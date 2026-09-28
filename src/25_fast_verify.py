"""
25_fast_verify.py

Fast O(n) verification of final submission files.
Uses hash-based/set-based membership checks — no nested loops.
"""

from pathlib import Path
import csv
import re
import sys
from collections import Counter

ROOT = Path(r"C:\Users\adity\hackathon")
MATCHING_OUTPUT = ROOT / "outputs" / "final" / "matching_results.tsv"
CANDIDATE_OUTPUT = ROOT / "outputs" / "final" / "candidate_pairs.tsv"
BACKUP_OUTPUT = ROOT / "outputs" / "final" / "matching_results_v6_original.tsv"

EXPECTED_ROWS = 1_732_544
EXPECTED_HEADER_MATCHING = ["source1_entity_id", "matched_entity_ids"]
EXPECTED_HEADER_CANDIDATE = ["source1_entity_id", "candidate_entity_ids"]

S1_RE = re.compile(r"^S1-\d+$")
S2_RE = re.compile(r"^S2-\d+$")
S3_RE = re.compile(r"^S3-\d+$")
VALID_RE = re.compile(r"^S[123]-\d+$")


def main():
    print("=" * 78)
    print("FAST FINAL SUBMISSION VERIFICATION")
    print("=" * 78)

    # ------------------------------------------------------------
    # 1. File existence
    # ------------------------------------------------------------
    print("\n[1] File existence and readability...")
    for f in [MATCHING_OUTPUT, CANDIDATE_OUTPUT]:
        if not f.exists():
            print(f"  FAIL: {f} does not exist")
            sys.exit(1)
        if not f.is_file():
            print(f"  FAIL: {f} is not a regular file")
            sys.exit(1)
        print(f"  OK: {f.name} ({f.stat().st_size:,} bytes)")

    # ------------------------------------------------------------
    # 2. Headers
    # ------------------------------------------------------------
    print("\n[2] Header check...")
    with open(MATCHING_OUTPUT, "r", encoding="utf-8") as f:
        m_header = next(csv.reader(f, delimiter="\t"))
    with open(CANDIDATE_OUTPUT, "r", encoding="utf-8") as f:
        c_header = next(csv.reader(f, delimiter="\t"))

    m_header_ok = m_header == EXPECTED_HEADER_MATCHING
    c_header_ok = c_header == EXPECTED_HEADER_CANDIDATE
    print(f"  matching_results.tsv: {m_header} {'PASS' if m_header_ok else 'FAIL'}")
    print(f"  candidate_pairs.tsv:  {c_header} {'PASS' if c_header_ok else 'FAIL'}")

    # ------------------------------------------------------------
    # 3. Row counts + S1 uniqueness + malformed IDs (single pass)
    # ------------------------------------------------------------
    print("\n[3] Row counts, S1 uniqueness, malformed IDs (single pass)...")

    m_s1_ids = []
    m_s1_set = set()
    m_dup_s1 = 0
    m_malformed_s1 = 0
    m_malformed_match = 0
    m_total_match_pairs = 0
    m_s2_matches = 0
    m_s3_matches = 0
    m_zero_match = 0
    m_with_match = 0
    m_dup_match_pairs = 0
    m_has_pandas_index = False

    with open(MATCHING_OUTPUT, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for i, row in enumerate(reader, 1):
            if len(row) < 2:
                continue
            sid = row[0]
            m_s1_ids.append(sid)

            # Check for pandas index (numeric first column)
            if i <= 3 and sid.isdigit():
                m_has_pandas_index = True

            # S1 uniqueness
            if sid in m_s1_set:
                m_dup_s1 += 1
            else:
                m_s1_set.add(sid)

            # Malformed S1
            if not S1_RE.match(sid):
                m_malformed_s1 += 1

            # Match analysis
            matches_str = row[1].strip()
            if not matches_str:
                m_zero_match += 1
            else:
                m_with_match += 1
                match_ids = [x.strip() for x in matches_str.split(",") if x.strip()]
                m_total_match_pairs += len(match_ids)

                # Check for duplicate matches
                match_set = set()
                for mid in match_ids:
                    if mid in match_set:
                        m_dup_match_pairs += 1
                    else:
                        match_set.add(mid)

                    # Malformed match ID
                    if not VALID_RE.match(mid):
                        m_malformed_match += 1

                    # S2 vs S3
                    if S2_RE.match(mid):
                        m_s2_matches += 1
                    elif S3_RE.match(mid):
                        m_s3_matches += 1

    m_rows = len(m_s1_ids)
    print(f"  matching_results.tsv data rows: {m_rows:,} (expected: {EXPECTED_ROWS:,})")
    print(f"  Unique S1 IDs: {len(m_s1_set):,}")
    print(f"  Duplicate S1 IDs: {m_dup_s1:,}")
    print(f"  Malformed S1 IDs: {m_malformed_s1:,}")
    print(f"  Malformed match IDs: {m_malformed_match:,}")
    print(f"  Pandas index column detected: {m_has_pandas_index}")

    # ------------------------------------------------------------
    # 4. Candidate pairs (single pass, build hash set)
    # ------------------------------------------------------------
    print("\n[4] Candidate pairs (single pass, building hash set)...")

    c_s1_ids = []
    c_s1_set = set()
    c_dup_s1 = 0
    c_malformed_s1 = 0
    c_has_pandas_index = False
    candidate_set = set()  # (sid, cid) pairs

    with open(CANDIDATE_OUTPUT, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for i, row in enumerate(reader, 1):
            if len(row) < 2:
                continue
            sid = row[0]
            c_s1_ids.append(sid)

            if i <= 3 and sid.isdigit():
                c_has_pandas_index = True

            if sid in c_s1_set:
                c_dup_s1 += 1
            else:
                c_s1_set.add(sid)

            if not S1_RE.match(sid):
                c_malformed_s1 += 1

            cands_str = row[1].strip()
            if cands_str:
                for cid in cands_str.split(","):
                    cid = cid.strip()
                    if cid:
                        candidate_set.add((sid, cid))

    c_rows = len(c_s1_ids)
    print(f"  candidate_pairs.tsv data rows: {c_rows:,} (expected: {EXPECTED_ROWS:,})")
    print(f"  Unique S1 IDs: {len(c_s1_set):,}")
    print(f"  Duplicate S1 IDs: {c_dup_s1:,}")
    print(f"  Malformed S1 IDs: {c_malformed_s1:,}")
    print(f"  Pandas index column detected: {c_has_pandas_index}")
    print(f"  Total candidate pairs: {len(candidate_set):,}")

    # ------------------------------------------------------------
    # 5. Match subset of candidates (O(n) hash lookup)
    # ------------------------------------------------------------
    print("\n[5] Match subset of candidates (O(n) hash lookup)...")

    matches_outside = 0
    with open(MATCHING_OUTPUT, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            sid = row[0]
            matches_str = row[1].strip()
            if matches_str:
                for mid in matches_str.split(","):
                    mid = mid.strip()
                    if mid and (sid, mid) not in candidate_set:
                        matches_outside += 1

    print(f"  Matches outside candidate set: {matches_outside:,}")

    # ------------------------------------------------------------
    # 6. Compare against backup
    # ------------------------------------------------------------
    print("\n[6] Compare against backup (matching_results_v6_original.tsv)...")

    if not BACKUP_OUTPUT.exists():
        print("  Backup file not found — skipping comparison")
    else:
        backup_pairs = set()
        with open(BACKUP_OUTPUT, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # skip header
            for row in reader:
                if len(row) >= 2:
                    sid = row[0]
                    matches_str = row[1].strip()
                    if matches_str:
                        for mid in matches_str.split(","):
                            mid = mid.strip()
                            if mid:
                                backup_pairs.add((sid, mid))

        # Current pairs
        current_pairs = set()
        with open(MATCHING_OUTPUT, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # skip header
            for row in reader:
                if len(row) >= 2:
                    sid = row[0]
                    matches_str = row[1].strip()
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
    print(f"  {'FINAL FILE:':<35} {MATCHING_OUTPUT}")
    print(f"  {'ROW COUNT:':<35} {m_rows:,} {'PASS' if m_rows == EXPECTED_ROWS else 'FAIL'}")
    print(f"  {'UNIQUE S1:':<35} {len(m_s1_set):,}")
    print(f"  {'S1 COVERAGE:':<35} {len(m_s1_set):,} / {EXPECTED_ROWS:,} {'PASS' if len(m_s1_set) == EXPECTED_ROWS else 'FAIL'}")
    print(f"  {'ZERO MATCH S1:':<35} {m_zero_match:,}")
    print(f"  {'TOTAL MATCH PAIRS:':<35} {m_total_match_pairs:,}")
    print(f"  {'S2 MATCHES:':<35} {m_s2_matches:,}")
    print(f"  {'S3 MATCHES:':<35} {m_s3_matches:,}")
    print(f"  {'DUPLICATE PAIRS:':<35} {m_dup_match_pairs + m_dup_s1:,}")
    print(f"  {'MALFORMED IDS:':<35} {m_malformed_s1 + m_malformed_match:,}")
    print(f"  {'MATCHES OUTSIDE CANDIDATES:':<35} {matches_outside:,}")
    print(f"  {'FORMAT STATUS:':<35} {'PASS' if (m_header_ok and c_header_ok and m_rows == EXPECTED_ROWS and len(m_s1_set) == EXPECTED_ROWS and m_dup_s1 == 0 and m_malformed_s1 == 0 and m_malformed_match == 0 and matches_outside == 0 and not m_has_pandas_index) else 'FAIL'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
