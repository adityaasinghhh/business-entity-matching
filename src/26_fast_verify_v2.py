"""
26_fast_verify_v2.py

Fast O(n) verification using pandas for vectorized I/O.
"""

from pathlib import Path
import re
import sys
import time

import pandas as pd

ROOT = Path(r"C:\Users\adity\hackathon")
MATCHING_OUTPUT = ROOT / "outputs" / "final" / "matching_results.tsv"
CANDIDATE_OUTPUT = ROOT / "outputs" / "final" / "candidate_pairs.tsv"
BACKUP_OUTPUT = ROOT / "outputs" / "final" / "matching_results_v6_original.tsv"

EXPECTED_ROWS = 1_732_544

S1_RE = re.compile(r"^S1-\d+$")
S2_RE = re.compile(r"^S2-\d+$")
S3_RE = re.compile(r"^S3-\d+$")
VALID_RE = re.compile(r"^S[123]-\d+$")


def main():
    print("=" * 78)
    print("FAST FINAL SUBMISSION VERIFICATION (pandas)")
    print("=" * 78)

    t0 = time.time()

    # ------------------------------------------------------------
    # 1. Read files with pandas (fast)
    # ------------------------------------------------------------
    print("\n[1] Reading files...")
    m = pd.read_csv(MATCHING_OUTPUT, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    print(f"  matching_results.tsv: {len(m):,} rows in {time.time()-t0:.1f}s")

    t1 = time.time()
    c = pd.read_csv(CANDIDATE_OUTPUT, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    print(f"  candidate_pairs.tsv:  {len(c):,} rows in {time.time()-t1:.1f}s")

    # ------------------------------------------------------------
    # 2. Headers
    # ------------------------------------------------------------
    print("\n[2] Header check...")
    m_header_ok = list(m.columns) == ["source1_entity_id", "matched_entity_ids"]
    c_header_ok = list(c.columns) == ["source1_entity_id", "candidate_entity_ids"]
    print(f"  matching_results.tsv: {list(m.columns)} {'PASS' if m_header_ok else 'FAIL'}")
    print(f"  candidate_pairs.tsv:  {list(c.columns)} {'PASS' if c_header_ok else 'FAIL'}")

    # ------------------------------------------------------------
    # 3. Row counts
    # ------------------------------------------------------------
    print("\n[3] Row counts...")
    print(f"  matching_results.tsv: {len(m):,} data rows (expected: {EXPECTED_ROWS:,})")
    print(f"  candidate_pairs.tsv:  {len(c):,} data rows (expected: {EXPECTED_ROWS:,})")

    # ------------------------------------------------------------
    # 4. S1 uniqueness and duplicates
    # ------------------------------------------------------------
    print("\n[4] S1 uniqueness and duplicates...")
    m_unique_s1 = m["source1_entity_id"].nunique()
    c_unique_s1 = c["source1_entity_id"].nunique()
    m_dup_s1 = len(m) - m_unique_s1
    c_dup_s1 = len(c) - c_unique_s1
    print(f"  matching_results.tsv unique S1: {m_unique_s1:,}")
    print(f"  candidate_pairs.tsv unique S1:  {c_unique_s1:,}")
    print(f"  matching_results.tsv duplicate S1: {m_dup_s1:,}")
    print(f"  candidate_pairs.tsv duplicate S1:  {c_dup_s1:,}")

    # ------------------------------------------------------------
    # 5. Malformed IDs
    # ------------------------------------------------------------
    print("\n[5] Malformed ID check...")
    m_s1_valid = m["source1_entity_id"].str.match(S1_RE)
    c_s1_valid = c["source1_entity_id"].str.match(S1_RE)
    m_malformed_s1 = (~m_s1_valid).sum()
    c_malformed_s1 = (~c_s1_valid).sum()
    print(f"  matching_results.tsv malformed S1: {m_malformed_s1:,}")
    print(f"  candidate_pairs.tsv malformed S1:  {c_malformed_s1:,}")

    # Check for pandas index (numeric first column)
    m_has_index = m["source1_entity_id"].str.match(r"^\d+$").any()
    c_has_index = c["source1_entity_id"].str.match(r"^\d+$").any()
    print(f"  matching_results.tsv pandas index: {m_has_index}")
    print(f"  candidate_pairs.tsv pandas index:  {c_has_index}")

    # ------------------------------------------------------------
    # 6. Match analysis (vectorized)
    # ------------------------------------------------------------
    print("\n[6] Match analysis...")

    # Count matches per S1
    m["match_count"] = m["matched_entity_ids"].apply(
        lambda x: len([z for z in x.split(",") if z.strip()]) if x.strip() else 0
    )
    m_zero_match = (m["match_count"] == 0).sum()
    m_with_match = (m["match_count"] > 0).sum()
    m_total_match_pairs = m["match_count"].sum()

    print(f"  S1 with zero matches: {m_zero_match:,}")
    print(f"  S1 with >=1 match:   {m_with_match:,}")
    print(f"  Total match pairs:    {m_total_match_pairs:,}")

    # Count S2 vs S3 matches
    print("\n[7] S2 vs S3 match counts...")
    s2_count = 0
    s3_count = 0
    malformed_count = 0
    dup_pairs = 0

    # Process in chunks for memory
    chunk_size = 100_000
    for start in range(0, len(m), chunk_size):
        chunk = m.iloc[start:start+chunk_size]
        for matches_str in chunk["matched_entity_ids"]:
            if not matches_str.strip():
                continue
            ids = [x.strip() for x in matches_str.split(",") if x.strip()]
            seen = set()
            for mid in ids:
                if mid in seen:
                    dup_pairs += 1
                else:
                    seen.add(mid)
                if S2_RE.match(mid):
                    s2_count += 1
                elif S3_RE.match(mid):
                    s3_count += 1
                else:
                    malformed_count += 1

    print(f"  S2 matches: {s2_count:,}")
    print(f"  S3 matches: {s3_count:,}")
    print(f"  Malformed match IDs: {malformed_count:,}")
    print(f"  Duplicate match pairs: {dup_pairs:,}")

    # ------------------------------------------------------------
    # 8. Match subset of candidates (hash-based O(n))
    # ------------------------------------------------------------
    print("\n[8] Match subset of candidates (hash-based)...")

    # Build candidate set: dict sid -> set of cands
    t2 = time.time()
    cand_dict = {}
    for _, row in c.iterrows():
        sid = row["source1_entity_id"]
        cands_str = row["candidate_entity_ids"].strip()
        if cands_str:
            cands = frozenset(x.strip() for x in cands_str.split(",") if x.strip())
            cand_dict[sid] = cands

    print(f"  Built candidate dict: {len(cand_dict):,} S1 in {time.time()-t2:.1f}s")

    # Check matches
    t3 = time.time()
    matches_outside = 0
    for _, row in m.iterrows():
        sid = row["source1_entity_id"]
        matches_str = row["matched_entity_ids"].strip()
        if not matches_str:
            continue
        cands = cand_dict.get(sid, frozenset())
        for mid in matches_str.split(","):
            mid = mid.strip()
            if mid and mid not in cands:
                matches_outside += 1

    print(f"  Matches outside candidate set: {matches_outside:,}")
    print(f"  Check time: {time.time()-t3:.1f}s")

    # ------------------------------------------------------------
    # 9. Compare against backup
    # ------------------------------------------------------------
    print("\n[9] Compare against backup...")
    if not BACKUP_OUTPUT.exists():
        print("  Backup not found — skipping")
    else:
        t4 = time.time()
        backup = pd.read_csv(BACKUP_OUTPUT, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
        backup_pairs = set()
        for _, row in backup.iterrows():
            sid = row["source1_entity_id"]
            matches_str = row["matched_entity_ids"].strip()
            if matches_str:
                for mid in matches_str.split(","):
                    mid = mid.strip()
                    if mid:
                        backup_pairs.add((sid, mid))

        current_pairs = set()
        for _, row in m.iterrows():
            sid = row["source1_entity_id"]
            matches_str = row["matched_entity_ids"].strip()
            if matches_str:
                for mid in matches_str.split(","):
                    mid = mid.strip()
                    if mid:
                        current_pairs.add((sid, mid))

        new_pairs = current_pairs - backup_pairs
        removed_pairs = backup_pairs - current_pairs

        print(f"  Backup unique pairs: {len(backup_pairs):,}")
        print(f"  Current unique pairs: {len(current_pairs):,}")
        print(f"  New pairs (in current, not in backup): {len(new_pairs):,}")
        print(f"  Removed pairs (in backup, not in current): {len(removed_pairs):,}")

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERIFICATION SUMMARY")
    print("=" * 78)

    format_pass = (
        m_header_ok
        and c_header_ok
        and len(m) == EXPECTED_ROWS
        and len(c) == EXPECTED_ROWS
        and m_dup_s1 == 0
        and c_dup_s1 == 0
        and m_malformed_s1 == 0
        and c_malformed_s1 == 0
        and malformed_count == 0
        and matches_outside == 0
        and not m_has_index
        and not c_has_index
    )

    print(f"  {'FINAL FILE:':<35} {MATCHING_OUTPUT}")
    print(f"  {'ROW COUNT:':<35} {len(m):,} {'PASS' if len(m) == EXPECTED_ROWS else 'FAIL'}")
    print(f"  {'UNIQUE S1:':<35} {m_unique_s1:,}")
    print(f"  {'S1 COVERAGE:':<35} {m_unique_s1:,} / {EXPECTED_ROWS:,} {'PASS' if m_unique_s1 == EXPECTED_ROWS else 'FAIL'}")
    print(f"  {'ZERO MATCH S1:':<35} {m_zero_match:,}")
    print(f"  {'TOTAL MATCH PAIRS:':<35} {m_total_match_pairs:,}")
    print(f"  {'S2 MATCHES:':<35} {s2_count:,}")
    print(f"  {'S3 MATCHES:':<35} {s3_count:,}")
    print(f"  {'DUPLICATE PAIRS:':<35} {dup_pairs + m_dup_s1 + c_dup_s1:,}")
    print(f"  {'MALFORMED IDS:':<35} {m_malformed_s1 + c_malformed_s1 + malformed_count:,}")
    print(f"  {'MATCHES OUTSIDE CANDIDATES:':<35} {matches_outside:,}")
    print(f"  {'FORMAT STATUS:':<35} {'PASS' if format_pass else 'FAIL'}")
    print("=" * 78)
    print(f"\nTotal verification time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
