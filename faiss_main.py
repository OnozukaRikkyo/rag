#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Patent XML extractor for ST.96 / ST.36 at scale (4M docs ready)

Features:
- Standard detection (ST.96 / ST.36)
- Latest content resolution (ST.96 amendments, ST.36 amended-claims)
- Formatting tag sanitation (com:U/Sup/Sub/Br/P removal/translation)
- Incremental runs (skip already-satisfied fields; merge outputs)
- Parallel processing, robust error handling, mirrored folder structure
"""

from __future__ import annotations

import pandas as pd
from pandas.io.sas.sas_constants import row_count_on_mix_page_offset_multiplier
from sympy.integrals.meijerint_doc import category

print("start")
import argparse
import json
import os
import re
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Set
import time
from lxml import etree

import psutil
import inspect
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures import wait, FIRST_COMPLETED
from tqdm import tqdm


def lineno():
    return inspect.currentframe().f_back.f_lineno

DEBUG = True

# =========================
#  CONFIG (no hard-coding)
# =========================
input_folder = '/mnt/eightthdd/similarity'  # 入力フォルダのパスを指定
read_data_folder = '/mnt/eightthdd/jsonl_data'  # 入力フォルダのパスを指定
output_folder = '/mnt/eightthdd/faiss_data'  # 出力フォルダのパスを指定
error_folder = '/mnt/eightthdd/error_faiss_data'
class_folder = '/mnt/eightthdd/classification'

@dataclass(frozen=True)
class Config:
    CSV = "1"

    INPUT_ROOT=Path(input_folder).resolve()
    READ_DATA_ROOT=Path(read_data_folder).resolve()
    OUTPUT_ROOT=Path(output_folder).resolve()
    ERRORS_ROOT=Path(error_folder).resolve()
    CLASS_ROOT=Path(class_folder).resolve()

    EXT_IN = "info.jsonl"
    READ_EXT = ".jsonl"
    EXT_OUT = ".txt"
    EXT_ERR = ".txt"

    # output one JSON object per line; one file per patent (jsonl extension)
    OUTPUT_EXT: str = ".jsonl"
    OUTPUT_INFO_NAME: str = "triples"

    # worker & parser tuning
    NUM_WORKERS: int = os.cpu_count() or 4
    XML_PARSER_RECOVER: bool = True
    XML_PARSER_HUGE: bool = True

    # incremental processing
    SKIP_IF_ALL_FIELDS_PRESENT: bool = True

    # error handling
    COPY_ORIGINAL_ON_ERROR: bool = True
    ERROR_LOG_NAME: str = "error.log"

    # versioning (increment when sanitizer/extractor logic changes)
    PIPELINE_VERSION: str = "1.0.0"

    ALL_FIELDS: Tuple[str, ...] = ("abstract", "claims", "description")  # extendable


# =========================
#  Utilities
# =========================

def build_config(args: argparse.Namespace) -> Config:
    # create initial config then override with CLI
    cfg = Config(
        INPUT_ROOT=Path(args.input).resolve(),
        OUTPUT_ROOT=Path(args.output).resolve(),
        ERRORS_ROOT=Path(args.errors).resolve(),
        NUM_WORKERS=args.workers,
        STRIP_MODE=args.strip_mode
    )
    return cfg

def ensure_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def rel_to_root(path: Path, root: Path) -> Path:
    path = path.resolve()
    root = root.resolve()
    return path.relative_to(root)

def out_path_with_new_name(cfg: Config, xml_path: Path) -> Path:
    rel = rel_to_root(xml_path, cfg.READ_DATA_ROOT)
    rel = rel.parent / cfg.OUTPUT_INFO_NAME
    out_rel = rel.with_suffix(cfg.OUTPUT_EXT)
    return (cfg.OUTPUT_ROOT / out_rel)

def get_output_root(cfg: Config) -> Path:
    return cfg.OUTPUT_ROOT

def out_path_from_input(cfg: Config, xml_path: Path) -> Path:
    rel = rel_to_root(xml_path, cfg.INPUT_ROOT)
    out_rel = rel.with_suffix(cfg.OUTPUT_EXT)
    return (cfg.OUTPUT_ROOT / out_rel)

def err_dir_for(cfg: Config, xml_path: Path) -> Path:
    rel = rel_to_root(xml_path, cfg.INPUT_ROOT)
    return (cfg.ERRORS_ROOT / rel.parent)

def write_error(cfg: Config, xml_path: Path, err: Exception):
    ed = err_dir_for(cfg, xml_path)
    ensure_dirs(ed)
    # copy original xml
    if cfg.COPY_ORIGINAL_ON_ERROR:
        try:
            shutil.copy2(xml_path, ed / xml_path.name)
        except Exception:
            print("Exception:", lineno())  # この行の番号が出力されます

    # write error log (append)
    log_path = ed / cfg.ERROR_LOG_NAME
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"=== ERROR: {xml_path} ===\n")
        f.write("".join(traceback.format_exception(type(err), err, err.__traceback__)))
        f.write("\n")

def load_existing_jsonl(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            line = f.readline()
            return json.loads(line) if line else None
    except Exception:
        return None

def write_jsonl_atomic(path: Path, obj: Dict):
    ensure_dirs(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(path)


import threading
MAX_WRITERS = 1  # 例: 同時書き込みは2本までに制限（環境に合わせて調整）
write_sem = threading.Semaphore(MAX_WRITERS)

def append_text_atomic(path: Path, text: str):
    global write_sem
    # 2) ディスク書き込みの直前だけ制限
    write_sem.acquire()
    try:
        path_parent = path.parent
        if not path_parent.is_dir():
            raise Exception(f"Path {path} is not a directory")
        ensure_dirs(path_parent)
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    finally:
        write_sem.release()
    return

def need_processing(existing: Optional[Dict], requested_fields: Set[str]) -> bool:
    if existing is None:
        return True
    if CFG.SKIP_IF_ALL_FIELDS_PRESENT:
        return False
    else: return True

def normalize_whitespace(text: str) -> str:
    if text is None:
        return ""
    # collapse multiple spaces, keep newlines
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join([ln for ln in lines if ln])

from lxml import etree
from typing import Dict, List, Set, Optional


# =========================
#  Main pipeline
# =========================
import calc_faiss
def process_one(path_tuple: Path, requested_fields: Set[str]) -> None:
    try:
        claim_path = path_tuple[0]
        ref_path = path_tuple[1]
        row_num = path_tuple[2]

        claim_text = process_get_text(claim_path)
        ref_text = process_get_text(ref_path)
        save_text_as_file(row_num, claim_text, ref_text)
        # sim = calc_faiss.calc_sim(claim_text, ref_text)
        # output_sim_to_df(sim, row_num)
    except Exception as e:
        print(path_tuple)
        print(e)

def save_text_as_file(row_num, text_a, text_b):

    dir_a = Path(f"data/text{CFG.CSV}_a")
    dir_b = Path(f"data/text{CFG.CSV}_b")
    ensure_dirs(dir_a)
    ensure_dirs(dir_b)

    # save text with a name of row number
    path_a = dir_a / f"{row_num}.txt"
    path_b = dir_b / f"{row_num}.txt"
    with open(path_a, "w", encoding="utf-8") as f:
        f.write(text_a)
    with open(path_b, "w", encoding="utf-8") as f:
        f.write(text_b)

def output_sim_to_df(sim, row_num):
    df = pd.read_csv("data/df_concat.csv")
    df.loc[row_num, "sim"] = sim
    df.to_csv("data/df_concat.csv", index=False)

def process_get_text(path_lib):
        text = path_lib.read_text(encoding="utf-8", errors="ignore")

        # jsonl parser
        info_dict = json.loads(text)

        source_path = get_jsonl_path_from_source(info_dict)
        claim_text = get_claim1_text(source_path)
        return claim_text

    #     if len(unique_class_list) == 0:
    #         raise Exception("No classification found")
    #
    #     outp = get_output_root(CFG)
    #     for class_str in unique_class_list:
    #         full_path = outp / (class_str + EXT_OUT)
    #         append_text_atomic(full_path, source_path)
    #
    # except Exception as e:
    #     write_error(CFG, xml_path, e)

def get_jsonl_path_from_source(class_dict_list):
    source_text = class_dict_list["source_path"]
    source_path = Path(source_text).resolve()
    # replace the top 2 parts of directory to CFG.READ_DATA_ROOT and change the extension to CFG.READ_EXT
    source_path = Path(str(source_path).replace("/mnt/onetsdd/raw_data", str(CFG.READ_DATA_ROOT)))
    source_path = source_path.with_suffix(CFG.READ_EXT)
    return source_path

def get_claim1_text(jsonl_path: Path) -> str:
        text = jsonl_path.read_text(encoding="utf-8", errors="ignore")
        data = json.loads(text)
        claims_text = data["fields"]["claims"][0]["text"]
        return claims_text

def walk_dir_files(root: Path) -> Iterable[Path]:
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False) and entry.name.endswith(CFG.EXT_IN):
                        yield Path(entry.path)
        except PermissionError:
            print("PermissionError:", lineno())  # この行の番号が出力されます
            continue  # 権限不足フォルダはスキップ

import subprocess
def find_directories(dir_name: str) -> str:
    # Example usage: find all directories under /path/to/search that contain "example" in their name
    result = subprocess.run(
        ["find", "//mnt/eightthdd/similarity", "-type", "d", "-name", dir_name],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        return line.strip()
    print(f"No directory named {dir_name} found.")
    return ""


def walk_class_files() -> Iterable[Path]:
    # open D.txt file in root directory
    csv_numer_str = CFG.CSV
    df = pd.read_csv(f"data/csv{csv_numer_str}_matches.csv")
    # set 1 to every raw in df1 of column number
    df["number"] = csv_numer_str
    # remove rows with himotuki is "[]"
    df = df[~df["matches_ref"].astype(str).str.strip().eq("[]")]
    df = df[~df["matches"].astype(str).str.strip().eq("[]")]

    df = df[df["category"].astype(str).str.strip().eq("Ax")]

    df.to_csv(f"data/df_concat{csv_numer_str}.csv", index=False)

    # loop with row in dataframe
    # for yield return row number
    for row_num, row in df.iterrows():
        try:
            claim_str = eval(row['matches'])[0]
            ref_str = eval(row['matches_ref'])[0]
            claim_path = Path(claim_str)
            ref_path = Path(ref_str)

            claim_path = claim_path / CFG.EXT_IN
            ref_path = ref_path / CFG.EXT_IN

            if not claim_path.exists():
                print(f"Json file in Claim path {claim_path} not found")
                continue
            if not ref_path.exists():
                print(f"Json file in Reference path {ref_path} not found")

            yield claim_path, ref_path, row_num
        except Exception as e:
            print(row)
            print(e)


if DEBUG:
    print("Debug")


def main():

    global CFG

    if DEBUG:
        num_workers_param = 1
    else:
        num_workers_param = (os.cpu_count() - 1)

    CFG = Config(
        NUM_WORKERS=num_workers_param
    )
    ensure_dirs(CFG.OUTPUT_ROOT)
    ensure_dirs(CFG.ERRORS_ROOT)
    wanted = set(CFG.ALL_FIELDS)

    # 未完了Futureの上限（大きすぎるとメモリ圧迫。ワーカー数の数倍が目安）
    MAX_OUTSTANDING = max(4, CFG.NUM_WORKERS * 4)

    processed = 0  # 実際に処理を投げて完了した件数
    skipped = 0  # スキップ（既出力済み等）
    submitted = 0  # 投入済み（未完了含む）
    # total_fixed = 4_347_010
    # number of dataframe rows
    #df = pd.read_csv("data/df_concat.csv")
    # get number of rows of df
    #rows = df.shape[0]
    total_fixed = 200000

    futures = set()

    # 進捗バー：総件数は固定値。スキップも完了も1件として進める
    pbar = tqdm(total=total_fixed, dynamic_ncols=True, desc="Processing", unit="file", smoothing=0)

    def drain_completed(nonblock=False):
        """完了済みFutureを回収してカウンタ更新。nonblock=True なら即時1回だけチェック。"""
        nonlocal processed
        if not futures:
            return

        # === ここだけ修正 ===
        if nonblock:
            # 即時チェック（timeout=0）
            done, _ = wait(futures, timeout=0, return_when=FIRST_COMPLETED)
        else:
            # 少なくとも1件完了するまで待機
            done, _ = wait(futures, return_when=FIRST_COMPLETED)

        # 同時に複数終わっている分を一気に取り切る
        if done:
            more_done, _ = wait(futures, timeout=0)
            done |= more_done
        # === 修正ここまで ===

        for f in done:
            futures.discard(f)
            try:
                f.result()  # 例外はここで発火
            except Exception as e:
                # どのXMLで失敗したかは process_one 内の write_error で記録される
                # ここでは潰して続行
                print("Exception:", lineno())  # この行の番号が出力されます
            processed += 1

    THREAD_EXEC = True

    if DEBUG:
        # 並列ではなく逐次処理
        # from tqdm import tqdm
        processed = 0
        skipped = 0
        # total_fixed = 4_347_010
        total_fixed = 347_010

        pbar = tqdm(total=total_fixed, desc="Processing (DEBUG sequential)", dynamic_ncols=True)

        for claim_ref_path_tuple in walk_class_files():
            # if not need_processing(existing, wanted):
            #     pbar.update(1)  # スキップでも1件前進
            #     continue

            try:
                process_one(claim_ref_path_tuple, wanted)  # 並列ではなく直接呼び出す
            except Exception as e:
                print("error", claim_ref_path_tuple)
                # write_error(CFG, json_path, e)

            processed += 1
            pbar.update(1)

        pbar.close()
        print(f"[DEBUG] Done. processed={processed}, skipped={skipped}, total_fixed={total_fixed}")

    elif THREAD_EXEC:

        # --- ここから本番スレッド版 ---
        with ThreadPoolExecutor(max_workers=getattr(CFG, "NUM_THREADS", CFG.NUM_WORKERS)) as ex:
            for json_path in walk_class_files():

                # スキップ判定を復活させるなら以下を有効化
                # if not need_processing(existing, wanted):
                #     skipped += 1
                #     pbar.update(1)
                #     continue

                pbar.update(1)  # スキップでも1件前進の見せ方を維持

                # 未完了 Future が多すぎる場合は間引いてメモリ圧を回避
                while len(futures) >= MAX_OUTSTANDING:
                    drain_completed(nonblock=False)  # ブロッキングで回収

                # 1ファイル＝1タスク
                fut = ex.submit(process_one, json_path, wanted)
                futures.add(fut)
                submitted += 1

                # つど少しだけ回収（非ブロッキング）
                drain_completed(nonblock=True)

            # 残りの全タスクを回収
            while futures:
                drain_completed(nonblock=False)

        # プログレスバーを閉じる（総数未満でも見た目を整える）
        pbar.close()

        print(f"Done. processed={processed}, skipped={skipped}, submitted={submitted}, total_fixed={total_fixed}")


if __name__ == "__main__":
    main()