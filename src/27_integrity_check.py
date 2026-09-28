"""Fast integrity check for matching_results.tsv"""
from pathlib import Path
import re

MATCHING_OUTPUT = Path(r"C:\Users\adity\hackathon\outputs\final\matching_results.tsv")

S1_RE = re.compile(r"^S1-\d+$")
S2_RE = re.compile(r"^S2-\d+$")
S3_RE = re.compile(r"^S3-\d+$")

print("=" * 78)
print("FAST INTEGRITY CHECK")
print("=" * 78)

# 1. File readable
print("\n[1] File readable...")
try:
    with open(MATCHING_OUTPUT, "r", encoding="utf-8") as f:
        first_line = f.readline()
    print("  PASS - file is readable")
except Exception as e:
    print(f"  FAIL - {e}")

# 2. Header
print("\n[2] Header check...")
with open(MATCHING_OUTPUT, "r", encoding="utf-8") as f:
    header = f.readline().strip()
parts = header.split("\t")
header_ok = len(parts) == 2 and parts[0] == "source1_entity_id" and parts[1] == "matched_entity_ids"
print(f"  Header: {repr(header)}")
print(f"  {'PASS' if header_ok else 'FAIL'}")

# 3. Row count
print("\n[3] Row count...")
with open(MATCHING_OUTPUT, "r", encoding="utf-8") as f:
    row_count = sum(1 for _ in f)
print(f"  Data rows: {row_count:,} (expected: 1,732,544)")
print(f"  {'PASS' if row_count == 1_732_544 else 'FAIL'}")

# 4-6. Sample-based checks (first 10K rows)
print("\n[4-6] Sample-based checks (first 10K rows)...")
s1_ids = set()
dup_s1 = 0
malformed_s1 = 0
malformed_match = 0
has_index = False
s2_count = 0
s3_count = 0
total_matches = 0
zero_match = 0

with open(MATCHING_OUTPUT, "r", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 10000:
            break
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        sid = parts[0]
        if i < 3 and sid.isdigit():
            has_index = True
        if sid in s1_ids:
            dup_s1 += 1
        else:
            s1_ids.add(sid)
        if not S1_RE.match(sid):
            malformed_s1 += 1
        matches_str = parts[1].strip()
        if not matches_str:
            zero_match += 1
        else:
            for mid in matches_str.split(","):
                mid = mid.strip()
                if mid:
                    total_matches += 1
                    if S2_RE.match(mid):
                        s2_count += 1
                    elif S3_RE.match(mid):
                        s3_count += 1
                    else:
                        malformed_match += 1

print(f"  Sample size: {min(i+1, 10000):,} rows")
print(f"  Duplicate S1 IDs: {dup_s1}")
print(f"  Malformed S1 IDs: {malformed_s1}")
print(f"  Malformed match IDs: {malformed_match}")
print(f"  Pandas index column: {has_index}")
print(f"  S2 matches: {s2_count:,}")
print(f"  S3 matches: {s3_count:,}")
print(f"  Total matches: {total_matches:,}")
print(f"  Zero-match S1: {zero_match:,}")

# 7. Filename
print("\n[7] Competition filename...")
print("  Current file: outputs/final/matching_results.tsv")
print("  Official validator expects: matching_results.tsv")

# Summary
print("\n" + "=" * 78)
all_pass = header_ok and row_count == 1_732_544 and dup_s1 == 0 and malformed_s1 == 0 and malformed_match == 0 and not has_index
if all_pass:
    print("READY FOR UPLOAD")
else:
    print("SOME CHECKS FAILED")
print("=" * 78)
