"""
24_verify_final_submission.py

Verify the final submission:
1. Total row count in matching_results.tsv (expected: 1,732,544)
2. Total row count in candidate_pairs.tsv (expected: 1,732,544)
3. Every S1 ID from test_source1.tsv appears exactly once
4. Header correctness
5. No duplicate S1 IDs
6. All matches are in candidate_pairs
7. Entity ID format validation
8. Total wall-clock time from checkpoint/log
"""

from pathlib import Path
import csv
import re
import time
from collections import Counter

ROOT = Path(r"C:\Users\adity\hackathon")
S1_FILE = ROOT / "data" / "raw" / "train" / "train" / "test" / "test" / "test_source1.tsv"
MATCHING_OUTPUT = ROOT / "outputs" / "final" / "matching_results.tsv"
CANDIDATE_OUTPUT = ROOT / "outputs" / "final" / "candidate_pairs.tsv"
CHECKPOINT_FILE = ROOT / "outputs" / "final" / "checkpoint.txt"

# Expected values
EXPECTED_S1_COUNT = 1_732_544
EXPECTED_HEADER_MATCHING = ["source1_entity_id", "matched_entity_ids"]
EXPECTED_HEADER_CANDIDATE = ["source1_entity_id", "candidate_entity_ids"]

# Entity ID patterns
S1_PATTERN = re.compile(r"^S1-\d+$")
S2_PATTERN = re.compile(r"^S2-\d+$")
S3_PATTERN = re.compile(r"^S3-\d+$")
VALID_ID_PATTERN = re.compile(r"^S[123]-\d+$")


def main():
    print("=" * 78)
    print("FINAL SUBMISSION VERIFICATION")
    print("=" * 78)

    # ------------------------------------------------------------
    # 1. File existence
    # ------------------------------------------------------------
    print("\n[1] File existence check...")
    for f in [MATCHING_OUTPUT, CANDIDATE_OUTPUT, S1_FILE]:
        if not f.exists():
            print(f"  FAIL: {f} does not exist")
            return
        print(f"  OK: {f.name} ({f.stat().st_size:,} bytes)")

    # ------------------------------------------------------------
    # 2. Row counts
    # ------------------------------------------------------------
    print("\n[2] Row counts...")
    with open(MATCHING_OUTPUT, "r", encoding="utf-8") as f:
        matching_header = next(csv.reader(f, delimiter="\t"))
        matching_rows = sum(1 for _ in f)

    with open(CANDIDATE_OUTPUT, "r", encoding="utf-8") as f:
        candidate_header = next(csv.reader(f, delimiter="\t"))
        candidate_rows = sum(1 for _ in f)

    print(f"  matching_results.tsv: {matching_rows:,} data rows (expected: {EXPECTED_S1_COUNT:,})")
    print(f"  candidate_pairs.tsv:  {candidate_rows:,} data rows (expected: {EXPECTED_S1_COUNT:,})")

    row_count_pass = matching_rows == EXPECTED_S1_COUNT and candidate_rows == EXPECTED_S1_COUNT
    print(f"  Row count: {'PASS' if row_count_pass else 'FAIL'}")

    # ------------------------------------------------------------
    # 3. Header check
    # ------------------------------------------------------------
    print("\n[3] Header check...")
    header_pass = True
    if matching_header != EXPECTED_HEADER_MATCHING:
        print(f"  FAIL: matching_results.tsv header = {matching_header}")
        header_pass = False
    else:
        print(f"  OK: matching_results.tsv header = {matching_header}")

    if candidate_header != EXPECTED_HEADER_CANDIDATE:
        print(f"  FAIL: candidate_pairs.tsv header = {candidate_header}")
        header_pass = False
    else:
        print(f"  OK: candidate_pairs.tsv header = {candidate_header}")

    # ------------------------------------------------------------
    # 4. S1 ID coverage and uniqueness
    # ------------------------------------------------------------
    print("\n[4] S1 ID coverage and uniqueness...")
    s1_ids = set()
    with open(S1_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if row:
                s1_ids.add(row[0])
    print(f"  Expected S1 IDs: {len(s1_ids):,}")

    output_s1_ids = []
    with open(MATCHING_OUTPUT, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if row:
                output_s1_ids.append(row[0])

    id_counter = Counter(output_s1_ids)
    duplicate_ids = {k: v for k, v in id_counter.items() if v > 1}
    missing_ids = s1_ids - set(output_s1_ids)
    extra_ids = set(output_s1_ids) - s1_ids

    print(f"  Output S1 IDs: {len(output_s1_ids):,}")
    print(f"  Unique S1 IDs: {len(id_counter):,}")
    print(f"  Duplicate S1 IDs: {len(duplicate_ids):,}")
    print(f"  Missing S1 IDs: {len(missing_ids):,}")
    print(f"  Extra S1 IDs: {len(extra_ids):,}")

    coverage_pass = (
        len(output_s1_ids) == len(s1_ids)
        and len(duplicate_ids) == 0
        and len(missing_ids) == 0
        and len(extra_ids) == 0
    )
    print(f"  Coverage: {'PASS' if coverage_pass else 'FAIL'}")

    # ------------------------------------------------------------
    # 5. Entity ID format validation
    # ------------------------------------------------------------
    print("\n[5] Entity ID format validation...")
    invalid_id_count = 0
    null_id_count = 0
    sample_invalid = []

    with open(MATCHING_OUTPUT, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for i, row in enumerate(reader, 1):
            if not row or len(row) < 2:
                continue
            sid = row[0]
            if not S1_PATTERN.match(sid):
                null_id_count += 1
                if len(sample_invalid) < 3:
                    sample_invalid.append((i, sid, "S1 ID format"))
            matches_str = row[1].strip() if len(row) > 1 else ""
            if matches_str:
                for mid in matches_str.split(","):
                    mid = mid.strip()
                    if mid and not VALID_ID_PATTERN.match(mid):
                        invalid_id_count += 1
                        if len(sample_invalid) < 3:
                            sample_invalid.append((i, mid, "match ID format"))

    print(f"  Invalid/malformed IDs: {invalid_id_count:,}")
    print(f"  Null S1 IDs: {null_id_count:,}")
    if sample_invalid:
        print(f"  Sample invalid: {sample_invalid}")

    format_pass = invalid_id_count == 0 and null_id_count == 0
    print(f"  Format: {'PASS' if format_pass else 'FAIL'}")

    # ------------------------------------------------------------
    # 6. Match ⊆ Candidates check
    # ------------------------------------------------------------
    print("\n[6] Match subset of candidates check...")
    # Load all candidates
    candidate_set = set()
    with open(CANDIDATE_OUTPUT, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if row and len(row) >= 2:
                sid = row[0]
                cands = row[1].strip()
                if cands:
                    for c in cands.split(","):
                        c = c.strip()
                        if c:
                            candidate_set.add((sid, c))

    # Check all matches are in candidates
    invalid_match_count = 0
    with open(MATCHING_OUTPUT, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if row and len(row) >= 2:
                sid = row[0]
                matches_str = row[1].strip()
                if matches_str:
                    for m in matches_str.split(","):
                        m = m.strip()
                        if m and (sid, m) not in candidate_set:
                            invalid_match_count += 1

    print(f"  Predicted matches outside candidate set: {invalid_match_count:,}")
    subset_pass = invalid_match_count == 0
    print(f"  Match subset: {'PASS' if subset_pass else 'FAIL'}")

    # ------------------------------------------------------------
    # 7. Checkpoint / wall-clock time
    # ------------------------------------------------------------
    print("\n[7] Checkpoint info...")
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            checkpoint_val = f.read().strip()
        print(f"  Last checkpoint: {checkpoint_val}")
        try:
            checkpoint_num = int(checkpoint_val)
            print(f"  Rows processed: {checkpoint_num:,}")
            print(f"  Remaining: {EXPECTED_S1_COUNT - checkpoint_num:,}")
        except ValueError:
            print(f"  (could not parse checkpoint value)")
    else:
        print("  No checkpoint file")

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERIFICATION SUMMARY")
    print("=" * 78)
    print(f"  {'Row count (matching):':<35} {'PASS' if matching_rows == EXPECTED_S1_COUNT else 'FAIL'} ({matching_rows:,} / {EXPECTED_S1_COUNT:,})")
    print(f"  {'Row count (candidate):':<35} {'PASS' if candidate_rows == EXPECTED_S1_COUNT else 'FAIL'} ({candidate_rows:,} / {EXPECTED_S1_COUNT:,})")
    print(f"  {'Header check:':<35} {'PASS' if header_pass else 'FAIL'}")
    print(f"  {'S1 ID coverage:':<35} {'PASS' if coverage_pass else 'FAIL'}")
    print(f"  {'Entity ID format:':<35} {'PASS' if format_pass else 'FAIL'}")
    print(f"  {'Match subset:':<35} {'PASS' if subset_pass else 'FAIL'}")
    print("=" * 78)

    all_pass = (
        row_count_pass
        and header_pass
        and coverage_pass
        and format_pass
        and subset_pass
    )
    print(f"\n  OVERALL: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
