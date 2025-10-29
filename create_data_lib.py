# -*- coding: utf-8 -*-
DEBUG = True
"""
faiss_topk_search.py — コーパスの全チャンクを FAISS にインデックス化し、
長文クエリ（チャンク化）から類似 Top-K 文書の (doc_id, score) を返す。
既存 calc_faiss.py の実装方針を最大限踏襲。

使い方（単一クエリ）
python faiss_topk_search.py \
  --corpus_dir /path/to/corpus_texts \
  --query /path/to/query.txt \
  --topk 10 \
  --max_tokens 2048 --stride_tokens 1536 \
  --per_chunk_topk 8 \
  --reduce_mode topk-mean --reduce_topk 4 \
  [--model_name sentence-transformers/all-MiniLM-L6-v2] \
  [--out_json results.json]

複数クエリ
python faiss_topk_search.py \
  --corpus_dir /path/to/corpus_texts \
  --query_dir /path/to/queries \
  --topk 10 --out_csv results.csv
"""
import os
import re
import json
import csv
import argparse
from collections import defaultdict
from typing import Callable, Generator, Iterable, List, Optional, Tuple
from pathlib import Path
import shutil
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("faiss が見つかりません。`pip install faiss-gpu` などでインストールしてください。") from e


# ===================== 1) 前処理・I/O =====================

def _prep_texts(texts: List[str]) -> List[str]:
    """軽量な前処理（空白正規化・トリム）。"""
    out = []
    for t in texts:
        t = "" if t is None else str(t)
        t = re.sub(r"\s+", " ", t).strip()
        out.append(t)
    return out

def list_txt_files(dir_path: str) -> List[str]:
    fs = [f for f in os.listdir(dir_path) if f.lower().endswith(".txt")]
    fs.sort(key=lambda x: (len(os.path.splitext(x)[0]), os.path.splitext(x)[0]))
    return fs

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def basename_wo_ext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


# ===================== 2) トークン化とチャンク化 =====================

def default_char_tokenize(s: str) -> List[str]:
    """フォールバック：1文字=1トークン。必要に応じて差し替え可。"""
    return list(s or "")

def chunk_by_tokens(
    text: str,
    tokenize_fn: Callable[[str], List[str]],
    max_tokens: int,
    stride_tokens: Optional[int] = None,
) -> List[str]:
    """トークン長ベースのスライディングウィンドウでチャンク化。"""
    if stride_tokens is None:
        stride_tokens = int(max_tokens * 0.75)

    toks = tokenize_fn(text)
    n = len(toks)
    if n == 0:
        return [""]

    chunks: List[str] = []
    start = 0
    while start < n:
        end = min(start + max_tokens, n)
        chunk = "".join(toks[start:end])
        chunks.append(chunk)
        if end == n:
            break
        start += stride_tokens
    return chunks


# ===================== 3) 埋め込み（Torch） =====================
from typing import List, Optional, Callable, Any
import numpy as np
import torch
import torch.nn as nn

def _default_char_tokenize(text: str) -> List[str]:
    return list(text or "")

class TorchTextEmbedder:
    """
    優先順:
      1) Sentence-Transformers（.encode）
      2) GoogleGenerativeAIEmbeddings（.embed_documents）
      3) ハッシュ平均（フォールバック）
    共通I/F:
      encode(texts: List[str], batch_size: int) -> np.ndarray[float32]
    """
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        dim: int = 768,
        vocab_size: int = 200_003,
        tokenize_fn: Callable[[str], List[str]] = _default_char_tokenize,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = dim
        self.vocab_size = vocab_size
        self.tokenize_fn = tokenize_fn

        self._backend: Any = None
        self._backend_kind: str = "hash"  # "st" | "google" | "hash"

        # まず明示指定がGoogle系だったらGoogleを試す
        if self._try_setup_google(model_name):
            self._backend_kind = "google"
        elif self._try_setup_st(model_name):
            self._backend_kind = "sentence_transformer"
        else:
            self._setup_hash_backend()

    # ---- Backends setup ----
    def _try_setup_st(self, model_name: Optional[str]) -> bool:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception:
            return False
        try:
            name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            self._backend = SentenceTransformer(name, device=self.device)
            # max_seq_length はモデル依存なのでむやみに拡張しない
            return True
        except Exception as e:
            print(f"⚠️ Sentence-Transformers 初期化失敗: {e}")
            self._backend = None
            return False

    def _try_setup_google(self, model_name: str) -> bool:
        try:
            # from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
            print("comment out : from langchain_google_genai import GoogleGenerativeAIEmbeddings")
        except Exception as e:
            print(f"⚠️ langchain_google_genai import 失敗: {e}")
            return False
        try:
            # GOOGLE_API_KEY は環境変数で設定しておく
            self._backend = GoogleGenerativeAIEmbeddings(model=model_name)
            # 注意: max_seq_length は外部制御できない → 呼び出し側でチャンク化して渡す
            return True
        except Exception as e:
            print(f"⚠️ GoogleGenerativeAIEmbeddings 初期化失敗: {e}")
            self._backend = None
            return False

    def _setup_hash_backend(self) -> None:
        g = torch.Generator().manual_seed(42)
        self._emb_vocab = int(self.vocab_size)  # ★ 前回の AttributeError の原因に対処
        self._emb = nn.Embedding(self._emb_vocab, self.dim)
        torch.nn.init.normal_(self._emb.weight, mean=0.0, std=0.02, generator=g)
        self._emb.to(self.device).eval()

    # ---- Public API ----
    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        texts = _prep_texts(texts)
        if not texts:
            return np.zeros((0, self._infer_dim()), dtype=np.float32)

        if self._backend is not None:
            if self._backend_kind == "sentence_transformer":
                return self._encode_st(texts, batch_size)
            elif self._backend_kind == "google":
                return self._encode_google(texts, batch_size)

        return self._encode_hash(texts, batch_size)

    # ---- encode implementations ----
    def _encode_st(self, texts: List[str], batch_size: int) -> np.ndarray:
        X = self._backend.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(X, dtype=np.float32)

    def _encode_google(self, texts: List[str], batch_size: int) -> np.ndarray:
        outs = []
        n = len(texts)
        for i in range(0, n, max(1, batch_size)):
            batch = texts[i:i+batch_size]
            X = self._backend.embed_documents(batch)  # -> List[List[float]]
            outs.append(np.asarray(X, dtype=np.float32))
        return np.vstack(outs)

    def _encode_hash(self, texts: List[str], batch_size: int) -> np.ndarray:
        outs = []
        for i in range(0, len(texts), max(1, batch_size)):
            sub = texts[i: i + batch_size]
            vecs = []
            for s in sub:
                toks = self.tokenize_fn(s) or [""]
                ids = [(hash(tok) % self._emb_vocab) for tok in toks]
                tt = torch.tensor(ids, dtype=torch.long, device=self.device)
                e = self._emb(tt)                  # (L, D)
                v = e.mean(dim=0, keepdim=True)    # (1, D)
                vecs.append(v)
            V = torch.cat(vecs, dim=0).detach().cpu().numpy().astype(np.float32)
            outs.append(V)
        return np.vstack(outs)

    def _infer_dim(self) -> int:
        if self._backend_kind == "st":
            try:
                return int(self._backend.get_sentence_embedding_dimension())
            except Exception:
                return int(self.dim)
        elif self._backend_kind == "google":
            try:
                X = self._backend.embed_documents(["test"])
                return int(len(X[0]))
            except Exception:
                return int(self.dim)
        else:
            return int(self.dim)


# ===================== 4) 類似度補助 =====================

def l2_normalize_inplace(x: np.ndarray):
    if x.size == 0:
        return
    faiss.normalize_L2(x)

def build_ip_index(d: int, use_gpu: bool = True):
    cpu_index = faiss.IndexFlatIP(d)

    # depending in torch.cuda
    use_gpu = True
    if use_gpu and torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, cpu_index)
    return cpu_index


# ===================== 5) コーパスのインデックス化 =====================

def build_corpus_index(
    corpus_dir: str,
    embedder: TorchTextEmbedder,
    tokenize_fn: Callable[[str], List[str]] = default_char_tokenize,
    max_tokens: int = 2048,
    stride_tokens: Optional[int] = None,
    emb_batch_size: int = 32,
    use_gpu_index: bool = True,
):
    """
    返り値:
      index: FAISS IndexFlatIP (GPU化されている場合あり)
      chunk_docids: np.ndarray[int32]  # ベクトルごとの doc 行番号（0..M-1）
      doc_bases: List[str]             # 行番号→ベース名（"1","2",...）
    """
    files = list_txt_files(corpus_dir)
    if not files:
        raise ValueError(f"コーパスが空です: {corpus_dir}")

    doc_bases: List[str] = [basename_wo_ext(f) for f in files]
    all_vecs: List[np.ndarray] = []
    chunk_docids: List[int] = []

    pbar = tqdm(total=len(files), dynamic_ncols=True, desc="Processing", unit="file", smoothing=0)

    # 1パスで順次エンコード（巨大コーパスでは適宜分割追加に変更可）
    for di, fname in enumerate(files):
        pbar.update(1)  # スキップでも1件前進
        path = os.path.join(corpus_dir, fname)
        text = read_text(path)
        chunks = chunk_by_tokens(text, tokenize_fn, max_tokens, stride_tokens)
        V = embedder.encode(chunks, batch_size=emb_batch_size)  # (mi, d)
        if V.size == 0:
            continue
        l2_normalize_inplace(V)
        all_vecs.append(V)
        chunk_docids.extend([di] * V.shape[0])

    if not all_vecs:
        raise ValueError("全コーパスで埋め込みが空でした。")

    X = np.vstack(all_vecs).astype(np.float32)     # (N, d)
    chunk_docids = np.asarray(chunk_docids, dtype=np.int32)
    index = build_ip_index(X.shape[1], use_gpu=use_gpu_index)
    index.add(X)
    return index, chunk_docids, doc_bases


# ===================== 6) クエリ検索（doc集約 topk） =====================

def aggregate_scores_per_doc(
    indices: np.ndarray,      # (Q, kq)
    dists: np.ndarray,        # (Q, kq)  内積=コサイン類似度（L2正規化済）
    chunk_docids: np.ndarray, # (N,)
    reduce_mode: str = "topk-mean",
    reduce_topk: int = 4,
) -> List[Tuple[int, float]]:
    """
    FAISS検索で得た (indices, dists) を doc 単位に集約し、(doc_id, score) を返す。
    """
    # doc -> 類似度のリスト
    scores_by_doc: defaultdict[int, List[float]] = defaultdict(list)
    Q, kq = dists.shape
    for qi in range(Q):
        for kj in range(kq):
            vec_idx = int(indices[qi, kj])
            if vec_idx < 0:
                continue
            doc_id = int(chunk_docids[vec_idx])
            scores_by_doc[doc_id].append(float(dists[qi, kj]))

    # 集約
    results: List[Tuple[int, float]] = []
    for doc_id, vals in scores_by_doc.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float32)
        if reduce_mode == "max":
            score = float(arr.max())
        elif reduce_mode == "mean":
            score = float(arr.mean())
        elif reduce_mode == "topk-mean":
            k = min(reduce_topk, arr.size)
            score = float(np.partition(arr, -k)[-k:].mean())
        else:
            raise ValueError(f"unknown reduce_mode: {reduce_mode}")
        results.append((doc_id, score))

    # スコア降順
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def search_topk_for_query(
    index,
    chunk_docids: np.ndarray,
    doc_bases: List[str],
    query_text: str,
    embedder: TorchTextEmbedder,
    tokenize_fn: Callable[[str], List[str]] = default_char_tokenize,
    max_tokens: int = 2048,
    stride_tokens: Optional[int] = None,
    emb_batch_size: int = 32,
    per_chunk_topk: int = 8,       # 各クエリチャンクで取得する近傍数
    reduce_mode: str = "topk-mean",
    reduce_topk: int = 4,
    topk: int = 10,                # doc 単位の最終 Top-K
) -> List[Tuple[str, float]]:
    # クエリをチャンク化→埋め込み→正規化
    q_chunks = chunk_by_tokens(query_text, tokenize_fn, max_tokens, stride_tokens)
    Q = embedder.encode(q_chunks, batch_size=emb_batch_size)  # (q, d)
    if Q.size == 0:
        return []
    l2_normalize_inplace(Q)

    # 近傍探索（IndexFlatIP）
    per_chunk_topk = max(1, per_chunk_topk)
    per_chunk_topk = min(per_chunk_topk, index.ntotal if hasattr(index, "ntotal") else per_chunk_topk)
    D, I = index.search(Q, per_chunk_topk)  # D: (q, kq)

    # doc 単位に集約
    doc_scores = aggregate_scores_per_doc(
        indices=I, dists=D,
        chunk_docids=chunk_docids,
        reduce_mode=reduce_mode,
        reduce_topk=reduce_topk,
    )
    # 上位 topk を doc_base 名とスコアで返す
    out: List[Tuple[str, float]] = []
    for doc_id, sc in doc_scores[:topk]:
        out.append((doc_bases[doc_id], float(sc)))
    return out


# ===================== 7) CLI =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_dir", required=True, help="コーパス .txt 群のディレクトリ（1.txt, 2.txt, ...）")
    gq = ap.add_mutually_exclusive_group(required=True)
    gq.add_argument("--query", help="単一クエリ .txt")
    gq.add_argument("--query_dir", help="複数クエリ .txt ディレクトリ")
    ap.add_argument("--model_name", default=None, help="sentence-transformers のモデル名（省略可）")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--stride_tokens", type=int, default=None)
    ap.add_argument("--emb_batch_size", type=int, default=32)
    ap.add_argument("--use_gpu_index", action="store_true", help="FAISSをGPU化（可能なら）")
    ap.add_argument("--per_chunk_topk", type=int, default=2048, help="クエリ各チャンクで取得する近傍数")
    ap.add_argument("--reduce_mode", choices=["max", "mean", "topk-mean"], default="topk-mean")
    ap.add_argument("--reduce_topk", type=int, default=4, help="reduce_mode=topk-mean の k")
    ap.add_argument("--topk", type=int, default=10, help="最終出力の doc Top-K")
    ap.add_argument("--out_json", default=None, help="単一クエリの結果を JSON で保存")
    ap.add_argument("--out_csv", default=None, help="複数クエリの結果を CSV で保存（query_id,doc_id,score,rank）")
    args = ap.parse_args()

    # 埋め込み器
    embedder = TorchTextEmbedder(model_name=args.model_name)

    # インデックス構築
    print("🔧 Building corpus index ...")
    index, chunk_docids, doc_bases = build_corpus_index(
        corpus_dir=args.corpus_dir,
        embedder=embedder,
        tokenize_fn=default_char_tokenize,
        max_tokens=args.max_tokens,
        stride_tokens=args.stride_tokens,
        emb_batch_size=args.emb_batch_size,
        use_gpu_index=args.use_gpu_index,
    )
    print(f"✅ Index built: vectors={getattr(index, 'ntotal', 'N/A')}, docs={len(doc_bases)}")

    # 単一クエリ
    if args.query:
        qtxt = read_text(args.query)
        results = search_topk_for_query(
            index=index,
            chunk_docids=chunk_docids,
            doc_bases=doc_bases,
            query_text=qtxt,
            embedder=embedder,
            tokenize_fn=default_char_tokenize,
            max_tokens=args.max_tokens,
            stride_tokens=args.stride_tokens,
            emb_batch_size=args.emb_batch_size,
            per_chunk_topk=args.per_chunk_topk,
            reduce_mode=args.reduce_mode,
            reduce_topk=args.reduce_topk,
            topk=args.topk,
        )
        print("=== Top-K ===")
        for rnk, (doc_id, score) in enumerate(results, 1):
            print(f"{rnk:>2}: doc={doc_id}  score={score:.6f}")

        if args.out_json:
            os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
            with open(args.out_json, "w", encoding="utf-8") as fw:
                json.dump(
                    [{"rank": i+1, "doc_id": d, "score": float(s)} for i, (d, s) in enumerate(results)],
                    fw, ensure_ascii=False, indent=2
                )
            print(f"📝 wrote: {args.out_json}")

    # 複数クエリ
    else:
        qfiles = list_txt_files(args.query_dir)
        if not qfiles:
            raise ValueError(f"クエリが空です: {args.query_dir}")

        rows: List[Tuple[str, str, float, int]] = []  # (query_id, doc_id, score, rank)

        pbar = tqdm(total=len(qfiles), dynamic_ncols=True, desc="query Processing", unit="file", smoothing=0)

        for i, qf in enumerate(qfiles):
            pbar.update(1)  # スキップでも1件前進
            if i < 3942:
                continue
            qid = basename_wo_ext(qf)
            qtxt = read_text(os.path.join(args.query_dir, qf))
            if i < 3942:
                if i >3940:
                    print(i, len(qtxt))
                continue
            if i < 8559:
                if i >8557:
                    print(i, len(qtxt))
                continue

            results = search_topk_for_query(
                index=index,
                chunk_docids=chunk_docids,
                doc_bases=doc_bases,
                query_text=qtxt,
                embedder=embedder,
                tokenize_fn=default_char_tokenize,
                max_tokens=args.max_tokens,
                stride_tokens=args.stride_tokens,
                emb_batch_size=args.emb_batch_size,
                per_chunk_topk=args.per_chunk_topk,
                reduce_mode=args.reduce_mode,
                reduce_topk=args.reduce_topk,
                topk=args.topk,
            )
            for rnk, (doc_id, score) in enumerate(results, 1):
                rows.append((qid, doc_id, float(score), rnk))

            df = pd.DataFrame(rows, columns=["q_id", "doc_id", "cos_sim", "rank"])
            filename = qid
            file_path = CFG.RESULT_RANK_ROOT / (filename + ".csv")
            df.to_csv(file_path,
                      index=False,  # DataFrameのインデックス(0, 1, 2...)をCSVに含めない
                      encoding='utf-8-sig')

        # # 出力
        # if args.out_csv:
        #     os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        #     with open(args.out_csv, "w", encoding="utf-8", newline="") as fw:
        #         w = csv.writer(fw)
        #         w.writerow(["query_id", "doc_id", "score", "rank"])
        #         for qid, did, sc, rnk in rows:
        #             w.writerow([qid, did, f"{sc:.6f}", rnk])
        #     print(f"📝 wrote: {args.out_csv}")
        # else:
        #     # 画面表示（各クエリごと）
        #     cur_qid = None
        #     for qid, did, sc, rnk in rows:
        #         if qid != cur_qid:
        #             cur_qid = qid
        #             print(f"\n=== Query: {qid} ===")
        #         print(f"{rnk:>2}: doc={did}  score={sc:.6f}")



# config.envファイルをルートに作って、APIキーを設定してください
# LANGSMITH_API_KEY=api_key
# GOOGLE_API_KEY=api_key

# .envファイルを読み込む
load_dotenv(dotenv_path="config.env")
OUTPUT_ROOT_DIR = os.environ.get("OUTPUT_ROOT")

# import os
os.environ["LANGSMITH_TRACING"] = "true"

class Config():

    CSV = "1" # データセットのCSVの番号
    REF_CONT = "ref" # 紐づき

    input_folder_name = "gr"
    output_folder_name = "text"
    output_folder_a = f'{OUTPUT_ROOT_DIR}/graph/csv{CSV}/{REF_CONT}/{output_folder_name}{CSV}_a'  # 出力フォルダのパスを指定
    output_folder_b = f'{OUTPUT_ROOT_DIR}/graph/csv{CSV}/{REF_CONT}/{output_folder_name}{CSV}_b'  # 出力フォルダのパスを指定

    REF_CONT = "cont" # 紐づかない、分類コード４桁が完全一致
    output_folder_cont = f'{OUTPUT_ROOT_DIR}/graph/csv{CSV}/{REF_CONT}/{output_folder_name}{CSV}_b'  # 出力フォルダのパスを指定


    OUTPUT_ROOT_A=Path(output_folder_a).resolve()
    OUTPUT_ROOT_B=Path(output_folder_b).resolve()
    OUTPUT_ROOT_CONT=Path(output_folder_cont).resolve()

    OUTPUT_ROOT_A.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT_B.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT_CONT.mkdir(parents=True, exist_ok=True)


    TEXT_A_DIR = f"{OUTPUT_ROOT_DIR}/graph/csv{CSV}/{REF_CONT}/{input_folder_name}{CSV}_a"
    TEXT_B_DIR = f"{OUTPUT_ROOT_DIR}/graph/csv{CSV}/{REF_CONT}/{input_folder_name}{CSV}_b"
    TEXT_A_ROOT = Path(TEXT_A_DIR).resolve()
    TEXT_B_ROOT = Path(TEXT_B_DIR).resolve()

    # ger all *.txt files in TEXT_A_DIR and TEXT_B_DIR and sort them
    TEXT_A_FILES = sorted(list(Path(TEXT_A_DIR).glob("*.pkl")))
    TEXT_B_FILES = sorted(list(Path(TEXT_B_DIR).glob("*.pkl")))

    # ベクトルデータとして登録する請求項１のテキストファイルを置く。ファイル名は公開文書ID
    corpus_folder = f"{OUTPUT_ROOT_DIR}/graph/csv{CSV}/corpus"
    CORPUS_ROOT = Path(corpus_folder).resolve()
    CORPUS_ROOT.mkdir(parents=True, exist_ok=True)

    query_folder = f"{OUTPUT_ROOT_DIR}/graph/csv{CSV}/query"
    QUERY_ROOT = Path(query_folder).resolve()
    QUERY_ROOT.mkdir(parents=True, exist_ok=True)

    out_csv = f"{OUTPUT_ROOT_DIR}/graph/csv{CSV}/results/topk_results_{CSV}.csv"
    # divide files into 3 parts

    result_rank_folder = f"{OUTPUT_ROOT_DIR}/graph/csv{CSV}/result_rank"
    RESULT_RANK_ROOT = Path(result_rank_folder).resolve()
    RESULT_RANK_ROOT.mkdir(parents=True, exist_ok=True)

    total_files = 6000
    num_divide = 3
    files_per_part = total_files // num_divide

CFG = Config()

CREATE_DATA = False

def configure_faiss():
    global CFG
    # FAISSの設定（必要に応じて調整）
    # faiss.omp_set_num_threads(4)
    # faiss.omp_set_num_threads(1)  # シングルスレッドにしたい場合
    # faiss.downcast_Index(faiss.IndexFlatIP(128)).metric_type = faiss.METRIC_L2  # 距離関数変更例

    df = pd.read_csv(f"{OUTPUT_ROOT_DIR}/graph/csv1/ref/df_concat1.csv")
    df_cont = pd.read_csv(f"{OUTPUT_ROOT_DIR}/graph/csv1/cont/df_concat1.csv")
    # dfのsyutugan列を取得
    claim_column = df["syutugan"].tolist()
    ref_column = df["himotuki"].tolist()
    ref_cont_column = df_cont["himotuki"].tolist()

    CFG.total_files = len(df)

    if CREATE_DATA:
    # データ作成用
        pbar = tqdm(total=CFG.total_files, dynamic_ncols=True, desc="Processing", unit="file", smoothing=0)

        for i in range(CFG.total_files):
            pbar.update(1)  # スキップでも1件前進

            p = CFG.OUTPUT_ROOT_B / f"{i}.txt"
            if not p.exists():
                continue
            target_file_name = ref_column[i]

            target = CFG.CORPUS_ROOT / f"{target_file_name}.txt"
            shutil.copy(p, target)

            p = CFG.OUTPUT_ROOT_A / f"{i}.txt"
            if not p.exists():
                continue
            target_file_name = claim_column[i]

            target = CFG.CORPUS_ROOT / f"{target_file_name}.txt"
            shutil.copy(p, target)

            target = CFG.QUERY_ROOT / f"{target_file_name}.txt"
            shutil.copy(p, target)

        #
        # for i in range(CFG.total_files):
        #     p = CFG.QUERY_ROOT / f"{i}.txt"
        #     if not p.exists():
        #         continue
        #     target_file_name = claim_column[i]
        #
        #     target = CFG.QUERY_ROOT / f"{target_file_name}.txt"
        #     shutil.copy(p, target)

if __name__ == "__main__":
    if CREATE_DATA:
        configure_faiss()
    import sys
    # 実行引数の設定
    sys.argv = [
        "top_k.py",
        "--corpus_dir", CFG.corpus_folder,
        "--query_dir", CFG.query_folder,
        "--topk", "100",
        "--max_tokens", "4096",
        "--stride_tokens", "512",
        "--out_csv", CFG.out_csv,
        # "--model_name", "models/text-embedding-004",
        # "--model_name", "models/gemini-embedding-001"
        "--model_name", "sentence-transformers/all-MiniLM-L6-v2"
    ]
    main()
