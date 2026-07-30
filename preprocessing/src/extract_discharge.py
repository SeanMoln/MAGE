"""
從 MIMIC-III 提取 Discharge Summary 100 筆，合併 ADMISSIONS 與 DIAGNOSES_ICD，
再執行 TextCleaner + SOAPParser 預處理。

輸入：
  - NOTEEVENTS.csv.gz   （篩選 Discharge summary）
  - ADMISSIONS.csv.gz   （入院基本資料）
  - DIAGNOSES_ICD.csv.gz（ICD-9 診斷碼，每筆 HADM_ID 最多取前 5 碼）

輸出：
  - preprocessing/data_raw/notes_discharge_100.csv          （合併後原始資料）
  - data/preprocessing/mimic_discharge_100_preprocessed.json（SOAP 預處理結果）
"""

import sys
import json
import random
import logging
import gzip
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from text_cleaner import TextCleaner
from soap_parser import SOAPParser

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── 路徑設定 ──────────────────────────────────────────────
BASE = Path(__file__).parent.parent.parent
MIMIC_DIR = BASE / "preprocessing/data_raw/physionet.org/files/mimiciii/1.4"
NOTEEVENTS_GZ  = MIMIC_DIR / "NOTEEVENTS.csv.gz"
ADMISSIONS_GZ  = MIMIC_DIR / "ADMISSIONS.csv.gz"
DIAGNOSES_GZ   = MIMIC_DIR / "DIAGNOSES_ICD.csv.gz"

def _dated_path(directory: Path, prefix: str, suffix: str) -> Path:
    """產生 prefix_YYYYMMDD_NNN.suffix 格式的路徑，同日期自動遞增序號。"""
    today = datetime.now().strftime("%Y%m%d")
    directory.mkdir(parents=True, exist_ok=True)
    existing = sorted(directory.glob(f"{prefix}_{today}_*.{suffix}"))
    if existing:
        last_num = int(existing[-1].stem.rsplit("_", 1)[-1])
        seq = last_num + 1
    else:
        seq = 1
    return directory / f"{prefix}_{today}_{seq:03d}.{suffix}"

OUTPUT_CSV  = BASE / "preprocessing/data_raw/notes_discharge_100.csv"   # 固定（供檢查用）

RANDOM_SEED = 42
SAMPLE_SIZE = 100
MIN_WORD_COUNT = 150   # Discharge summary 通常很長，設高一點

# ADMISSIONS 要保留的欄位
ADM_COLS = [
    "HADM_ID", "SUBJECT_ID",
    "ADMISSION_TYPE", "ADMISSION_LOCATION", "DISCHARGE_LOCATION",
    "INSURANCE", "MARITAL_STATUS", "ETHNICITY",
    "DIAGNOSIS",           # 入院主要診斷（文字）
    "HOSPITAL_EXPIRE_FLAG",
]


def load_admissions() -> pd.DataFrame:
    logger.info("📂 讀取 ADMISSIONS...")
    df = pd.read_csv(ADMISSIONS_GZ, usecols=ADM_COLS)
    logger.info(f"   {len(df)} 筆入院記錄，{df['HADM_ID'].nunique()} 個唯一 HADM_ID")
    return df


def load_diagnoses_icd(top_n: int = 5) -> pd.DataFrame:
    """每筆住院只保留 SEQ_NUM <= top_n 的主要診斷碼，聚合成列表。"""
    logger.info("📂 讀取 DIAGNOSES_ICD...")
    df = pd.read_csv(DIAGNOSES_GZ, usecols=["HADM_ID", "SEQ_NUM", "ICD9_CODE"])
    df = df[df["SEQ_NUM"] <= top_n].copy()
    df["ICD9_CODE"] = df["ICD9_CODE"].astype(str).str.strip()
    grouped = (
        df.sort_values(["HADM_ID", "SEQ_NUM"])
          .groupby("HADM_ID")["ICD9_CODE"]
          .apply(list)
          .reset_index()
          .rename(columns={"ICD9_CODE": "icd9_codes"})
    )
    logger.info(f"   聚合完成，{len(grouped)} 筆 HADM_ID 有診斷碼")
    return grouped


def sample_discharge_notes(sample_size: int = SAMPLE_SIZE, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """水庫抽樣：從 NOTEEVENTS 隨機取 sample_size 筆 Discharge summary。"""
    logger.info("📂 讀取 NOTEEVENTS（Discharge summary 篩選）...")
    rng = random.Random(random_seed)
    reservoir = []
    eligible = 0

    with gzip.open(NOTEEVENTS_GZ, "rt", encoding="utf-8") as f:
        for chunk in pd.read_csv(
            f,
            chunksize=10000,
            usecols=["ROW_ID", "SUBJECT_ID", "HADM_ID",
                     "CHARTDATE", "CATEGORY", "DESCRIPTION", "ISERROR", "TEXT"],
        ):
            # 篩選：Discharge summary、無錯誤標記、有文字、字數足夠
            mask = (
                (chunk["CATEGORY"] == "Discharge summary") &
                (chunk["ISERROR"].isna() | (chunk["ISERROR"] == 0)) &
                chunk["TEXT"].notna() &
                (chunk["TEXT"].str.split().str.len() >= MIN_WORD_COUNT)
            )
            filtered = chunk[mask]

            for _, row in filtered.iterrows():
                eligible += 1
                record = row.to_dict()
                if len(reservoir) < sample_size:
                    reservoir.append(record)
                else:
                    idx = rng.randint(0, eligible - 1)
                    if idx < sample_size:
                        reservoir[idx] = record

            if eligible % 10000 == 0 and eligible > 0:
                logger.info(f"   已掃描到 {eligible} 筆符合條件...")

    logger.info(f"   總符合筆數: {eligible}，抽取: {len(reservoir)}")
    return pd.DataFrame(reservoir)


def build_soap_text(soap: dict) -> str:
    parts = []
    for section in ["S", "O", "A", "P"]:
        content = soap.get(section, "").strip()
        if content:
            parts.append(f"{section}: {content}")
    return "\n\n".join(parts)


def main(sample_size=None, random_seed=None, min_word_count=None, icd_top_n=5):
    sample_size    = sample_size    or SAMPLE_SIZE
    random_seed    = random_seed    if random_seed is not None else RANDOM_SEED
    min_word_count = min_word_count or MIN_WORD_COUNT

    logger.info("=" * 65)
    logger.info(f"MARS-PMB  Discharge Summary {sample_size} 筆  提取 + 合併 + 預處理")
    logger.info(f"  seed={random_seed}  min_words={min_word_count}  icd_top_n={icd_top_n}")
    logger.info("=" * 65)

    # ── 1. 抽樣 Discharge summary ─────────────────────────────
    notes_df = sample_discharge_notes(sample_size=sample_size, random_seed=random_seed)

    # ── 2. 合併 ADMISSIONS ─────────────────────────────────────
    admissions_df = load_admissions()
    merged = notes_df.merge(admissions_df, on=["HADM_ID", "SUBJECT_ID"], how="left")
    logger.info(f"\n合併 ADMISSIONS 後: {len(merged)} 筆")
    unmatched = merged["ADMISSION_TYPE"].isna().sum()
    if unmatched:
        logger.info(f"  ⚠️  {unmatched} 筆無對應 ADMISSIONS 記錄（HADM_ID 不匹配）")

    # ── 3. 合併 DIAGNOSES_ICD ──────────────────────────────────
    diag_df = load_diagnoses_icd(top_n=icd_top_n)
    merged = merged.merge(diag_df, on="HADM_ID", how="left")
    merged["icd9_codes"] = merged["icd9_codes"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    logger.info(f"合併 DIAGNOSES_ICD 後: {len(merged)} 筆")
    no_diag = (merged["icd9_codes"].str.len() == 0).sum()
    if no_diag:
        logger.info(f"  ⚠️  {no_diag} 筆無對應 ICD 診斷碼")

    # ── 4. 儲存合併後原始資料（日期流水號命名）──────────────────
    out_csv  = _dated_path(BASE / "preprocessing/data_raw", "discharge",              "csv")
    out_json = _dated_path(BASE / "data/preprocessing",    "discharge_preprocessed", "json")

    save_df = merged.copy()
    save_df["icd9_codes"] = save_df["icd9_codes"].apply(lambda x: "|".join(x))
    save_df.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info(f"\n💾 合併原始資料已儲存: {out_csv}")

    # ── 5. 預處理：TextCleaner + SOAPParser ────────────────────
    logger.info("\n🔧 開始預處理（TextCleaner + SOAPParser）...")
    cleaner = TextCleaner()
    parser  = SOAPParser()

    results = []
    stats = {"S": 0, "O": 0, "A": 0, "P": 0, "quality_fail": 0}

    for idx, row in merged.iterrows():
        raw_text = row.get("TEXT", "") or ""
        if not raw_text.strip():
            logger.info(f"  [{idx+1:3d}] 空白文本，跳過")
            continue

        # TextCleaner
        clean_result = cleaner.clean_text(raw_text)
        cleaned = clean_result.get("cleaned_text", "")
        if not clean_result.get("quality_passed", True):
            stats["quality_fail"] += 1

        # SOAPParser
        soap = parser.parse(cleaned)
        soap_text = build_soap_text(soap)

        # 附加入院上下文（追加在 SOAP 文末，供系統參考）
        icd_list = row.get("icd9_codes", [])
        adm_diag = row.get("DIAGNOSIS", "")
        adm_type = row.get("ADMISSION_TYPE", "")

        context_lines = []
        if adm_type:
            context_lines.append(f"Admission type: {adm_type}")
        if adm_diag:
            context_lines.append(f"Admission diagnosis: {adm_diag}")
        if icd_list:
            context_lines.append(f"ICD-9 codes: {', '.join(icd_list)}")

        if context_lines:
            soap_text = soap_text + "\n\nContext: " + " | ".join(context_lines)

        has = {k: bool(soap.get(k, "").strip()) for k in ["S", "O", "A", "P"]}
        for k in ["S", "O", "A", "P"]:
            if has[k]:
                stats[k] += 1

        results.append(soap_text)
        found = "".join(k for k in ["S", "O", "A", "P"] if has[k])
        i = len(results)
        logger.info(
            f"  [{i:3d}] 區段:[{found}]  "
            f"清理前:{len(raw_text):5d}→清理後:{len(cleaned):5d}  "
            f"SOAP:{len(soap_text):4d}字元  "
            f"ICD:{len(icd_list)}碼"
        )

    # ── 6. 儲存 JSON ────────────────────────────────────────────
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("")
    logger.info("=" * 65)
    logger.info(f"✅ 完成！輸出：{out_json}（{len(results)} 筆）")
    logger.info("")
    logger.info("SOAP 區段覆蓋率：")
    for k in ["S", "O", "A", "P"]:
        pct = stats[k] / len(results) * 100 if results else 0
        logger.info(f"  {k}: {stats[k]}/{len(results)} ({pct:.1f}%)")
    if stats["quality_fail"]:
        logger.info(f"品質未通過: {stats['quality_fail']} 筆")
    logger.info("=" * 65)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="MIMIC-III Discharge summary 預處理")
    ap.add_argument("--seed",         type=int,   default=RANDOM_SEED,    help="隨機種子（預設 42）")
    ap.add_argument("--sample",       type=int,   default=SAMPLE_SIZE,    help="抽取筆數（預設 100）")
    ap.add_argument("--min_words",    type=int,   default=MIN_WORD_COUNT,  help="最小字數門檻（預設 150）")
    ap.add_argument("--icd_top_n",    type=int,   default=5,              help="ICD-9 碼最多取前 N 碼（預設 5）")
    args = ap.parse_args()

    main(
        sample_size=args.sample,
        random_seed=args.seed,
        min_word_count=args.min_words,
        icd_top_n=args.icd_top_n,
    )
