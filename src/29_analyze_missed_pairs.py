"""
29_analyze_missed_pairs.py

Fast analysis of missed pairs to find blocking improvements.
Identifies patterns in missed GT pairs and tests new blocking rules.
"""

from pathlib import Path
import re
import time
from collections import defaultdict, Counter

import pandas as pd

ROOT = Path(r"C:\Users\adity\hackathon")
RAW_TRAIN = ROOT / "data" / "raw" / "train" / "train"
PROCESSED = ROOT / "data" / "processed"

S1_FILE = RAW_TRAIN / "train_source1.tsv"
S2_FILE = RAW_TRAIN / "train_source2.tsv"
S3_FILE = RAW_TRAIN / "train_source3.tsv"
GT_FILE = RAW_TRAIN / "train_ground_truth.tsv"

N_S1_SAMPLES = 100_000

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
NUMBER_RE = re.compile(r"\d{1,8}")


def normalize(x):
    if x is None or pd.isna(x):
        return ""
    x = str(x).lower().strip()
    if not x:
        return ""
    x = URL_RE.sub(" ", x)
    x = NON_WORD_RE.sub(" ", x)
    return SPACE_RE.sub(" ", x).strip()


def tokens(x):
    return [t for t in x.split() if len(t) >= 2]


def prefix(x, n):
    z = re.sub(r"\s+", "", x)
    return z[:n] if len(z) >= n else ""


def suffix(x, n):
    z = re.sub(r"\s+", "", x)
    return z[-n:] if len(z) >= n else ""


def numbers(x):
    return set(NUMBER_RE.findall(x))


def main():
    print("=" * 78)
    print("MISSED PAIR ANALYSIS")
    print("=" * 78)

    # Load S1 sample
    print("\n[1] Loading S1 sample...")
    s1 = pd.read_csv(S1_FILE, sep="\t", dtype=str, nrows=N_S1_SAMPLES,
                     keep_default_na=False, na_filter=False)
    s1_ids = set(s1["entity_id"].astype(str))
    print(f"  S1 entities: {len(s1_ids):,}")

    # Load ground truth
    print("\n[2] Loading ground truth...")
    gt = pd.read_csv(GT_FILE, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    gt = gt[gt["source1_entity_id"].isin(s1_ids)]

    gt_lookup = {}
    for row in gt.itertuples(index=False):
        s1_id = row.source1_entity_id
        raw = row.matched_entity_ids.strip()
        if raw:
            matches = {x.strip() for x in raw.split(",") if x.strip()}
        else:
            matches = set()
        gt_lookup[s1_id] = matches

    total_gt_pairs = sum(len(v) for v in gt_lookup.values())
    print(f"  GT pairs: {total_gt_pairs:,}")

    # Load blocking candidates
    print("\n[3] Loading blocking candidates...")
    s2_cand = pd.read_csv(PROCESSED / "blocking_pilot_s1_s2_v4.csv", sep=",", dtype=str,
                         keep_default_na=False, na_filter=False)
    s3_cand = pd.read_csv(PROCESSED / "blocking_pilot_s1_s3_v4.csv", sep=",", dtype=str,
                         keep_default_na=False, na_filter=False)
    s2_cand = s2_cand[s2_cand["source1_entity_id"].isin(s1_ids)]
    s3_cand = s3_cand[s3_cand["source1_entity_id"].isin(s1_ids)]

    s2_cand_dict = {}
    for row in s2_cand.itertuples(index=False):
        s1_id = str(row.source1_entity_id)
        raw = str(row.candidate_entity_ids)
        if raw:
            cands = {x.strip() for x in raw.split("|") if x.strip()}
        else:
            cands = set()
        s2_cand_dict[s1_id] = cands

    s3_cand_dict = {}
    for row in s3_cand.itertuples(index=False):
        s1_id = str(row.source1_entity_id)
        raw = str(row.candidate_entity_ids)
        if raw:
            cands = {x.strip() for x in raw.split("|") if x.strip()}
        else:
            cands = set()
        s3_cand_dict[s1_id] = cands

    all_cand_dict = {}
    for s1_id in s1_ids:
        cands = set()
        cands.update(s2_cand_dict.get(s1_id, set()))
        cands.update(s3_cand_dict.get(s1_id, set()))
        all_cand_dict[s1_id] = cands

    # Identify missed pairs
    print("\n[4] Identifying missed pairs...")
    missed_pairs = []  # (s1_id, gt_match)
    recovered_pairs = []

    for s1_id, gt_matches in gt_lookup.items():
        if not gt_matches:
            continue
        cands = all_cand_dict.get(s1_id, set())
        for m in gt_matches:
            if m in cands:
                recovered_pairs.append((s1_id, m))
            else:
                missed_pairs.append((s1_id, m))

    print(f"  Recovered pairs: {len(recovered_pairs):,}")
    print(f"  Missed pairs: {len(missed_pairs):,}")
    print(f"  Blocking recall: {len(recovered_pairs) / total_gt_pairs:.4%}")

    # Load S1, S2, S3 records for missed pairs
    print("\n[5] Loading records for missed pairs...")

    # Get unique S1 and S2/S3 IDs needed
    missed_s1_ids = set(x[0] for x in missed_pairs)
    missed_s2_ids = set(x[1] for x in missed_pairs if x[1].startswith("S2-"))
    missed_s3_ids = set(x[1] for x in missed_pairs if x[1].startswith("S3-"))

    print(f"  Unique S1 IDs with misses: {len(missed_s1_ids):,}")
    print(f"  Unique S2 IDs missed: {len(missed_s2_ids):,}")
    print(f"  Unique S3 IDs missed: {len(missed_s3_ids):,}")

    # Load S1 records
    s1_records = {}
    for row in pd.read_csv(S1_FILE, sep="\t", dtype=str, chunksize=50000,
                           keep_default_na=False, na_filter=False):
        for r in row.itertuples(index=False):
            eid = str(r.entity_id)
            if eid in missed_s1_ids:
                s1_records[eid] = {
                    "name": str(r.business_name),
                    "address": str(r.business_address),
                    "country": str(r.country),
                }
        if len(s1_records) >= len(missed_s1_ids):
            break

    # Load S2 records
    s2_records = {}
    for row in pd.read_csv(S2_FILE, sep="\t", dtype=str, chunksize=50000,
                           keep_default_na=False, na_filter=False):
        for r in row.itertuples(index=False):
            eid = str(r.entity_id)
            if eid in missed_s2_ids:
                s2_records[eid] = {
                    "name": str(r.business_name),
                    "address": str(r.business_address),
                    "country": str(r.country),
                }
        if len(s2_records) >= len(missed_s2_ids):
            break

    # Load S3 records
    s3_records = {}
    for row in pd.read_csv(S3_FILE, sep="\t", dtype=str, chunksize=50000,
                           keep_default_na=False, na_filter=False):
        for r in row.itertuples(index=False):
            eid = str(r.entity_id)
            if eid in missed_s3_ids:
                s3_records[eid] = {
                    "name": str(r.business_name),
                    "address": str(r.business_address),
                    "country": str(r.country),
                }
        if len(s3_records) >= len(missed_s3_ids):
            break

    print(f"  Loaded S1 records: {len(s1_records):,}")
    print(f"  Loaded S2 records: {len(s2_records):,}")
    print(f"  Loaded S3 records: {len(s3_records):,}")

    # Analyze missed pairs
    print("\n[6] Analyzing missed pairs...")

    # Normalize all records
    for eid, rec in s1_records.items():
        rec["name_norm"] = normalize(rec["name"])
        rec["address_norm"] = normalize(rec["address"])
        rec["country_norm"] = rec["country"].lower().strip()
        rec["tokens"] = tokens(rec["name_norm"])
        rec["address_tokens"] = tokens(rec["address_norm"])
        rec["address_numbers"] = numbers(rec["address_norm"])

    for eid, rec in s2_records.items():
        rec["name_norm"] = normalize(rec["name"])
        rec["address_norm"] = normalize(rec["address"])
        rec["country_norm"] = rec["country"].lower().strip()
        rec["tokens"] = tokens(rec["name_norm"])
        rec["address_tokens"] = tokens(rec["address_norm"])
        rec["address_numbers"] = numbers(rec["address_norm"])

    for eid, rec in s3_records.items():
        rec["name_norm"] = normalize(rec["name"])
        rec["address_norm"] = normalize(rec["address"])
        rec["country_norm"] = rec["country"].lower().strip()
        rec["tokens"] = tokens(rec["name_norm"])
        rec["address_tokens"] = tokens(rec["address_norm"])
        rec["address_numbers"] = numbers(rec["address_norm"])

    # Analyze each missed pair
    stats = {
        "total": 0,
        "same_exact_name": 0,
        "same_name_diff_country": 0,
        "name_token_overlap_50": 0,
        "name_token_overlap_30": 0,
        "same_address_number": 0,
        "same_country": 0,
        "address_token_overlap": 0,
        "name_prefix_5": 0,
        "name_suffix_4": 0,
        "name_last_token": 0,
        "name_token_overlap_any": 0,
    }

    for s1_id, gt_id in missed_pairs:
        if s1_id not in s1_records:
            continue
        if gt_id.startswith("S2-") and gt_id not in s2_records:
            continue
        if gt_id.startswith("S3-") and gt_id not in s3_records:
            continue

        s1_rec = s1_records[s1_id]
        if gt_id.startswith("S2-"):
            gt_rec = s2_records.get(gt_id, {})
        else:
            gt_rec = s3_records.get(gt_id, {})

        if not gt_rec:
            continue

        stats["total"] += 1

        s1_name = s1_rec["name_norm"]
        gt_name = gt_rec["name_norm"]
        s1_country = s1_rec["country_norm"]
        gt_country = gt_rec["country_norm"]

        # Exact name match
        if s1_name == gt_name and s1_name:
            stats["same_exact_name"] += 1

        # Same name but different country
        if s1_name == gt_name and s1_name and s1_country != gt_country:
            stats["same_name_diff_country"] += 1

        # Token overlap
        s1_tokens = set(s1_rec["tokens"])
        gt_tokens = set(gt_rec["tokens"])
        if s1_tokens and gt_tokens:
            overlap = len(s1_tokens & gt_tokens) / max(len(s1_tokens), len(gt_tokens))
            if overlap >= 0.5:
                stats["name_token_overlap_50"] += 1
            if overlap >= 0.3:
                stats["name_token_overlap_30"] += 1
            if overlap > 0:
                stats["name_token_overlap_any"] += 1

        # Address number
        if s1_rec["address_numbers"] and gt_rec["address_numbers"]:
            if s1_rec["address_numbers"] & gt_rec["address_numbers"]:
                stats["same_address_number"] += 1

        # Same country
        if s1_country == gt_country and s1_country:
            stats["same_country"] += 1

        # Address token overlap
        s1_at = set(s1_rec["address_tokens"])
        gt_at = set(gt_rec["address_tokens"])
        if s1_at and gt_at:
            if s1_at & gt_at:
                stats["address_token_overlap"] += 1

        # Prefix/suffix
        if prefix(s1_name, 5) and prefix(s1_name, 5) == prefix(gt_name, 5):
            stats["name_prefix_5"] += 1
        if suffix(s1_name, 4) and suffix(s1_name, 4) == suffix(gt_name, 4):
            stats["name_suffix_4"] += 1
        if s1_rec["tokens"] and gt_rec["tokens"]:
            if s1_rec["tokens"][-1] == gt_rec["tokens"][-1]:
                stats["name_last_token"] += 1

    print(f"\n  Total missed pairs analyzed: {stats['total']:,}")
    print(f"  Same exact name: {stats['same_exact_name']:,} ({stats['same_exact_name']/stats['total']:.2%})")
    print(f"  Same name diff country: {stats['same_name_diff_country']:,}")
    print(f"  Name token overlap >= 50%: {stats['name_token_overlap_50']:,} ({stats['name_token_overlap_50']/stats['total']:.2%})")
    print(f"  Name token overlap >= 30%: {stats['name_token_overlap_30']:,} ({stats['name_token_overlap_30']/stats['total']:.2%})")
    print(f"  Name token overlap any: {stats['name_token_overlap_any']:,} ({stats['name_token_overlap_any']/stats['total']:.2%})")
    print(f"  Same address number: {stats['same_address_number']:,} ({stats['same_address_number']/stats['total']:.2%})")
    print(f"  Same country: {stats['same_country']:,} ({stats['same_country']/stats['total']:.2%})")
    print(f"  Address token overlap: {stats['address_token_overlap']:,} ({stats['address_token_overlap']/stats['total']:.2%})")
    print(f"  Name prefix 5 match: {stats['name_prefix_5']:,} ({stats['name_prefix_5']/stats['total']:.2%})")
    print(f"  Name suffix 4 match: {stats['name_suffix_4']:,} ({stats['name_suffix_4']/stats['total']:.2%})")
    print(f"  Name last token match: {stats['name_last_token']:,} ({stats['name_last_token']/stats['total']:.2%})")

    # What rules would recover these?
    print("\n[7] What new blocking rules would recover?")
    recoverable = stats["name_token_overlap_any"]
    print(f"  Any token overlap: {recoverable:,} ({recoverable/stats['total']:.2%})")
    recoverable = stats["address_token_overlap"]
    print(f"  Address token overlap: {recoverable:,} ({recoverable/stats['total']:.2%})")
    recoverable = stats["name_prefix_5"]
    print(f"  Name prefix 5: {recoverable:,} ({recoverable/stats['total']:.2%})")
    recoverable = stats["name_suffix_4"]
    print(f"  Name suffix 4: {recoverable:,} ({recoverable/stats['total']:.2%})")
    recoverable = stats["name_last_token"]
    print(f"  Name last token: {recoverable:,} ({recoverable/stats['total']:.2%})")

    # Combined recovery
    print("\n[8] Combined recovery potential...")
    any_recover = 0
    for s1_id, gt_id in missed_pairs:
        if s1_id not in s1_records:
            continue
        if gt_id.startswith("S2-") and gt_id not in s2_records:
            continue
        if gt_id.startswith("S3-") and gt_id not in s3_records:
            continue

        s1_rec = s1_records[s1_id]
        if gt_id.startswith("S2-"):
            gt_rec = s2_records.get(gt_id, {})
        else:
            gt_rec = s3_records.get(gt_id, {})

        if not gt_rec:
            continue

        s1_name = s1_rec["name_norm"]
        gt_name = gt_rec["name_norm"]

        # Check various rules
        if s1_name == gt_name and s1_name:
            any_recover += 1
            continue

        s1_tokens = set(s1_rec["tokens"])
        gt_tokens = set(gt_rec["tokens"])
        if s1_tokens and gt_tokens:
            overlap = len(s1_tokens & gt_tokens) / max(len(s1_tokens), len(gt_tokens))
            if overlap >= 0.3:
                any_recover += 1
                continue

        if prefix(s1_name, 5) and prefix(s1_name, 5) == prefix(gt_name, 5):
            any_recover += 1
            continue

        if suffix(s1_name, 4) and suffix(s1_name, 4) == suffix(gt_name, 4):
            any_recover += 1
            continue

        if s1_rec["tokens"] and gt_rec["tokens"]:
            if s1_rec["tokens"][-1] == gt_rec["tokens"][-1]:
                any_recover += 1
                continue

        s1_at = set(s1_rec["address_tokens"])
        gt_at = set(gt_rec["address_tokens"])
        if s1_at and gt_at:
            if s1_at & gt_at:
                any_recover += 1
                continue

        if s1_rec["address_numbers"] and gt_rec["address_numbers"]:
            if s1_rec["address_numbers"] & gt_rec["address_numbers"]:
                any_recover += 1
                continue

    print(f"  Pairs recoverable by any rule: {any_recover:,} ({any_recover/len(missed_pairs):.2%})")
    print(f"  New blocking recall: {(len(recovered_pairs) + any_recover) / total_gt_pairs:.4%}")

    print("\n" + "=" * 78)
    print("ANALYSIS COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
