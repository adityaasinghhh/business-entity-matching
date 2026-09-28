"""
diagnose_batch2_s2_lookup.py
=============================
Isolates the exact failing batch (S1 rows offset 2500..4999, i.e. the
second CHUNK_SIZE=2500 chunk) and checks its S2 candidate IDs against
BOTH the DuckDB cache and the raw current test_source2.tsv directly,
bypassing the full 1.7M-row pipeline entirely.

This does NOT modify any output files. Safe to run repeatedly.

Usage:
    conda activate entitymatch
    cd C:\\Users\\adity\\hackathon\\src
    python diagnose_batch2_s2_lookup.py
"""
from pathlib import Path
import csv
import re
import time
import os

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from rapidfuzz import process, fuzz

ROOT = Path(r"C:\Users\adity\hackathon")
DATA = ROOT / "data"
RAW_TEST = DATA / "raw" / "train" / "train" / "test" / "test"
PROCESSED = DATA / "processed"

S1_FILE = RAW_TEST / "test_source1.tsv"
S2_FILE = RAW_TEST / "test_source2.tsv"
S2_CANDIDATES = PROCESSED / "blocking_test_s1_s2_v6.tsv"
CACHE_DB = PROCESSED / "test_source_lookup.duckdb"

# The failing batch, per the traceback ("S1 batch starting 2,500").
# Change these if a different offset fails next time.
BATCH_START = 2500
BATCH_SIZE = 2500


def qpath(path):
    return str(path).replace("'", "''")


def parse_ids(cell):
    return [z.strip() for z in str(cell).split(",") if z.strip()]


def main():
    print("=" * 72)
    print(f"ISOLATING BATCH: S1 rows [{BATCH_START}:{BATCH_START + BATCH_SIZE}]")
    print("=" * 72)

    # --- 1. Read exactly this slice from S1 and the S2 candidate file ---
    s1_slice = pd.read_csv(
        S1_FILE, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
        skiprows=range(1, BATCH_START + 1), nrows=BATCH_SIZE,
    )
    s2_cand_slice = pd.read_csv(
        S2_CANDIDATES, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
        skiprows=range(1, BATCH_START + 1), nrows=BATCH_SIZE,
    )

    print(f"S1 rows read              : {len(s1_slice):,}")
    print(f"S2 candidate rows read    : {len(s2_cand_slice):,}")

    # --- 2. Confirm row-for-row S1 <-> candidate-file alignment for this slice ---
    s1_ids = s1_slice["entity_id"].astype(str).reset_index(drop=True)
    cand_s1_ids = s2_cand_slice["source1_entity_id"].astype(str).reset_index(drop=True)
    aligned = s1_ids.equals(cand_s1_ids)
    print(f"S1 id <-> candidate-file source1_entity_id aligned : {aligned}")
    if not aligned:
        mismatches = (s1_ids != cand_s1_ids)
        first_bad = mismatches.idxmax()
        print(f"  First mismatch at slice-relative row {first_bad}:")
        print(f"    S1 entity_id            = {s1_ids.iloc[first_bad]}")
        print(f"    candidate source1_entity_id = {cand_s1_ids.iloc[first_bad]}")
        print("  -> S1 file and blocking file are OUT OF ROW-SYNC for this batch.")
        print("     This alone would explain lookup failures. Stopping here.")
        return

    # --- 3. Parse candidate IDs for this slice ---
    candidate_lists = s2_cand_slice["candidate_entity_ids"].map(parse_ids)
    all_ids = sorted({cid for lst in candidate_lists for cid in lst})
    print(f"Unique S2 candidate IDs requested in this batch : {len(all_ids):,}")
    if not all_ids:
        print("  No candidate IDs at all in this slice — nothing to look up. "
              "(If the real run reported requested_s2 > 0, re-check the offset above.)")
        return
    print(f"  First 10 example IDs: {all_ids[:10]}")

    # --- 4. Check membership in DuckDB cache ---
    con = duckdb.connect(str(CACHE_DB), read_only=True)
    ids_df = pd.DataFrame({"entity_id": all_ids})
    con.register("_ids", ids_df)
    try:
        found_in_cache = con.execute("""
            SELECT i.entity_id FROM _ids i
            JOIN s2 s ON s.entity_id = i.entity_id
        """).fetchdf()["entity_id"].tolist()
    finally:
        con.unregister("_ids")

    print(f"\nFound in DuckDB `s2` table          : {len(found_in_cache):,}/{len(all_ids):,}")

    # --- 5. Check membership directly in the raw CURRENT test_source2.tsv ---
    con.register("_ids", ids_df)
    try:
        found_in_raw = con.execute(f"""
            SELECT i.entity_id FROM _ids i
            JOIN read_csv('{qpath(S2_FILE)}', delim='\t', header=true,
                          quote='"', escape='"', nullstr='') s
              ON CAST(s.entity_id AS VARCHAR) = i.entity_id
        """).fetchdf()["entity_id"].tolist()
    finally:
        con.unregister("_ids")

    print(f"Found directly in current test_source2.tsv : {len(found_in_raw):,}/{len(all_ids):,}")

    missing_from_cache = set(all_ids) - set(found_in_cache)
    missing_from_raw = set(all_ids) - set(found_in_raw)

    print(f"\nMissing from DuckDB cache : {len(missing_from_cache):,}")
    if missing_from_cache:
        print(f"  Example missing-from-cache IDs : {sorted(missing_from_cache)[:10]}")
    print(f"Missing from raw source   : {len(missing_from_raw):,}")
    if missing_from_raw:
        print(f"  Example missing-from-raw IDs   : {sorted(missing_from_raw)[:10]}")

    print("\n" + "=" * 72)
    if not missing_from_raw and not missing_from_cache:
        print("CONCLUSION: every requested ID exists in both the raw source and the "
              "cache. The earlier failure was likely NOT a data problem — re-check "
              "for a code-level bug (e.g. fetch_records dtype/whitespace handling, "
              "or a DuckDB connection/register issue specific to repeated calls).")
    elif missing_from_raw and not missing_from_cache:
        print("CONCLUSION: IDs are in the cache but not the current raw file — "
              "the cache was built from a DIFFERENT/OLDER version of "
              "test_source2.tsv than what's on disk now. Rebuild the cache.")
    elif missing_from_raw and missing_from_cache:
        print("CONCLUSION: these candidate IDs genuinely do not exist in the "
              "CURRENT test_source2.tsv at all. The V6 blocking file "
              "(blocking_test_s1_s2_v6.tsv) was generated against a different "
              "source than the one currently on disk, at least for this ID range. "
              "The blocking file needs to be regenerated from the current source, "
              "or the correct source file needs to be located and swapped in.")
    else:
        print("CONCLUSION: IDs exist in the raw file but NOT in the cache, despite "
              "row counts matching. This points to a subtle DuckDB cache build "
              "issue (e.g. partial/interrupted build, encoding mismatch). Rebuild "
              "the cache table from scratch.")
    print("=" * 72)

    con.close()


if __name__ == "__main__":
    main()