"""
33_fast_blocking.py

Lightfast blocking improvement using in-memory dictionaries.
Adds targeted candidates to existing V6 candidates.
No DuckDB indexes needed.
"""

from pathlib import Path
import re
import time
from collections import defaultdict

import pandas as pd

ROOT = Path(r"C:\Users\adity\hackathon")
RAW_TRAIN = ROOT / "data" / "raw" / "train" / "train"
PROCESSED = ROOT / "data" / "processed"

S1_FILE = RAW_TRAIN / "train_source1.tsv"
S2_FILE = RAW_TRAIN / "train_source2.tsv"
S3_FILE = RAW_TRAIN / "train_source3.tsv"
GT_FILE = RAW_TRAIN / "train_ground_truth.tsv"

N_S1_SAMPLES = 100_000
CHUNK_SIZE = 10_000

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


def numbers(x):
    return set(NUMBER_RE.findall(x))


def main():
    print("=" * 78)
    print("FAST BLOCKING IMPROVEMENT (100K sample)")
    print("=" * 78)

    t0 = time.time()

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

    # Load V6 blocking candidates
    print("\n[3] Loading V6 blocking candidates...")
    s2_cand = pd.read_csv(PROCESSED / "blocking_pilot_s1_s2_v4.csv", sep=",", dtype=str,
                         keep_default_na=False, na_filter=False)
    s3_cand = pd.read_csv(PROCESSED / "blocking_pilot_s1_s3_v4.csv", sep=",", dtype=str,
                         keep_default_na=False, na_filter=False)
    s2_cand = s2_cand[s2_cand["source1_entity_id"].isin(s1_ids)]
    s3_cand = s3_cand[s3_cand["source1_entity_id"].isin(s1_ids)]

    v6_cand = {}
    for row in s2_cand.itertuples(index=False):
        s1_id = str(row.source1_entity_id)
        raw = str(row.candidate_entity_ids)
        if raw:
            cands = {x.strip() for x in raw.split("|") if x.strip()}
        else:
            cands = set()
        v6_cand.setdefault(s1_id, set()).update(cands)

    for row in s3_cand.itertuples(index=False):
        s1_id = str(row.source1_entity_id)
        raw = str(row.candidate_entity_ids)
        if raw:
            cands = {x.strip() for x in raw.split("|") if x.strip()}
        else:
            cands = set()
        v6_cand.setdefault(s1_id, set()).update(cands)

    # Evaluate V6 baseline
    v6_recovered = 0
    for s1_id, gt_matches in gt_lookup.items():
        if not gt_matches:
            continue
        cands = v6_cand.get(s1_id, set())
        v6_recovered += len(cands & gt_matches)

    v6_recall = v6_recovered / total_gt_pairs if total_gt_pairs else 0
    print(f"  V6 blocking recall: {v6_recall:.4%}")
    print(f"  V6 recovered: {v6_recovered:,} / {total_gt_pairs:,}")

    # ============================================================
    # Build lightweight indexes for targeted rules
    # ============================================================
    print("\n[4] Building lightweight indexes...")

    # Load S1 records
    s1_records = {}
    for row in s1.itertuples(index=False):
        eid = str(row.entity_id)
        name_norm = normalize(row.business_name)
        addr_norm = normalize(row.business_address)
        country = str(row.country).lower().strip()
        s1_records[eid] = {
            "name_norm": name_norm,
            "addr_norm": addr_norm,
            "country": country,
            "tokens": set(tokens(name_norm)),
            "addr_tokens": set(t for t in addr_norm.split() if len(t) >= 3),
            "numbers": numbers(addr_norm),
        }

    # Load S2 records
    print("  Loading S2 records...")
    s2_records = {}
    for row in pd.read_csv(S2_FILE, sep="\t", dtype=str, chunksize=50000,
                           keep_default_na=False, na_filter=False):
        for r in row.itertuples(index=False):
            eid = str(r.entity_id)
            name_norm = normalize(r.business_name)
            addr_norm = normalize(r.business_address)
            country = str(r.country).lower().strip()
            s2_records[eid] = {
                "name_norm": name_norm,
                "addr_norm": addr_norm,
                "country": country,
                "tokens": set(tokens(name_norm)),
                "addr_tokens": set(t for t in addr_norm.split() if len(t) >= 3),
                "numbers": numbers(addr_norm),
            }
        if len(s2_records) >= 5_000_000:
            break

    # Load S3 records
    print("  Loading S3 records...")
    s3_records = {}
    for row in pd.read_csv(S3_FILE, sep="\t", dtype=str, chunksize=50000,
                           keep_default_na=False, na_filter=False):
        for r in row.itertuples(index=False):
            eid = str(r.entity_id)
            name_norm = normalize(r.business_name)
            addr_norm = normalize(r.business_address)
            country = str(r.country).lower().strip()
            s3_records[eid] = {
                "name_norm": name_norm,
                "addr_norm": addr_norm,
                "country": country,
                "tokens": set(tokens(name_norm)),
                "addr_tokens": set(t for t in addr_norm.split() if len(t) >= 3),
                "numbers": numbers(addr_norm),
            }
        if len(s3_records) >= 5_000_000:
            break

    print(f"  S1 records: {len(s1_records):,}")
    print(f"  S2 records: {len(s2_records):,}")
    print(f"  S3 records: {len(s3_records):,}")

    # ============================================================
    # Build targeted blocking indexes
    # ============================================================
    print("\n[5] Building targeted blocking indexes...")

    # Index 1: address_token -> entity_ids (for S2 and S3 separately)
    s2_addr_token_index = defaultdict(set)
    s3_addr_token_index = defaultdict(set)

    for eid, rec in s2_records.items():
        for tok in rec["addr_tokens"]:
            s2_addr_token_index[tok].add(eid)

    for eid, rec in s3_records.items():
        for tok in rec["addr_tokens"]:
            s3_addr_token_index[tok].add(eid)

    # Index 2: name_token -> entity_ids
    s2_name_token_index = defaultdict(set)
    s3_name_token_index = defaultdict(set)

    for eid, rec in s2_records.items():
        for tok in rec["tokens"]:
            s2_name_token_index[tok].add(eid)

    for eid, rec in s3_records.items():
        for tok in rec["tokens"]:
            s3_name_token_index[tok].add(eid)

    # Index 3: address_number -> entity_ids
    s2_number_index = defaultdict(set)
    s3_number_index = defaultdict(set)

    for eid, rec in s2_records.items():
        for num in rec["numbers"]:
            s2_number_index[num].add(eid)

    for eid, rec in s3_records.items():
        for num in rec["numbers"]:
            s3_number_index[num].add(eid)

    print(f"  S2 address token index: {len(s2_addr_token_index):,} tokens")
    print(f"  S3 address token index: {len(s3_addr_token_index):,} tokens")
    print(f"  S2 name token index: {len(s2_name_token_index):,} tokens")
    print(f"  S3 name token index: {len(s3_name_token_index):,} tokens")
    print(f"  S2 number index: {len(s2_number_index):,} numbers")
    print(f"  S3 number index: {len(s3_number_index):,} numbers")

    # ============================================================
    # Generate additional candidates
    # ============================================================
    print("\n[6] Generating additional candidates...")

    new_cand_s2 = defaultdict(set)
    new_cand_s3 = defaultdict(set)
    rules_applied = defaultdict(int)

    for s1_id, s1_rec in s1_records.items():
        existing = v6_cand.get(s1_id, set())

        # Rule 1: Address token overlap (any shared address token)
        # Only add if S1 has address tokens
        if s1_rec["addr_tokens"]:
            candidates = set()
            for tok in s1_rec["addr_tokens"]:
                candidates.update(s2_addr_token_index.get(tok, set()))
                candidates.update(s3_addr_token_index.get(tok, set()))
            # Remove existing and self
            candidates -= existing
            candidates.discard(s1_id)
            if candidates:
                new_cand_s2[s1_id].update(c for c in candidates if c.startswith("S2-"))
                new_cand_s3[s1_id].update(c for c in candidates if c.startswith("S3-"))
                rules_applied["addr_token_overlap"] += 1

        # Rule 2: Name token overlap (any shared name token) + country
        if s1_rec["tokens"]:
            candidates = set()
            for tok in s1_rec["tokens"]:
                # S2
                for eid in s2_name_token_index.get(tok, set()):
                    if s2_records[eid]["country"] == s1_rec["country"] and s1_rec["country"]:
                        candidates.add(eid)
                # S3
                for eid in s3_name_token_index.get(tok, set()):
                    if s3_records[eid]["country"] == s1_rec["country"] and s1_rec["country"]:
                        candidates.add(eid)
            candidates -= existing
            candidates.discard(s1_id)
            if candidates:
                new_cand_s2[s1_id].update(c for c in candidates if c.startswith("S2-"))
                new_cand_s3[s1_id].update(c for c in candidates if c.startswith("S3-"))
                rules_applied["name_token_overlap"] += 1

        # Rule 3: Address number + country
        if s1_rec["numbers"]:
            candidates = set()
            for num in s1_rec["numbers"]:
                for eid in s2_number_index.get(num, set()):
                    if s2_records[eid]["country"] == s1_rec["country"] and s1_rec["country"]:
                        candidates.add(eid)
                for eid in s3_number_index.get(num, set()):
                    if s3_records[eid]["country"] == s1_rec["country"] and s1_rec["country"]:
                        candidates.add(eid)
            candidates -= existing
            candidates.discard(s1_id)
            if candidates:
                new_cand_s2[s1_id].update(c for c in candidates if c.startswith("S2-"))
                new_cand_s3[s1_id].update(c for c in candidates if c.startswith("S3-"))
                rules_applied["addr_number_country"] += 1

    print(f"  Rules applied: {dict(rules_applied)}")
    print(f"  S1 with new S2 candidates: {len(new_cand_s2):,}")
    print(f"  S1 with new S3 candidates: {len(new_cand_s3):,}")

    # ============================================================
    # Evaluate new blocking recall
    # ============================================================
    print("\n[7] Evaluating new blocking recall...")

    new_recovered = 0
    new_total_candidates = 0
    max_candidates = 0

    for s1_id, gt_matches in gt_lookup.items():
        if not gt_matches:
            continue

        existing = v6_cand.get(s1_id, set())
        additional = new_cand_s2.get(s1_id, set()) | new_cand_s3.get(s1_id, set())
        all_cands = existing | additional

        new_total_candidates += len(all_cands)
        max_candidates = max(max_candidates, len(all_cands))

        new_recovered += len(all_cands & gt_matches)

    new_recall = new_recovered / total_gt_pairs if total_gt_pairs else 0
    avg_candidates = new_total_candidates / len(s1_records) if s1_records else 0

    print(f"\n  V6 blocking recall: {v6_recall:.4%}")
    print(f"  New blocking recall: {new_recall:.4%}")
    print(f"  Improvement: {new_recall - v6_recall:+.4%}")
    print(f"  V6 recovered: {v6_recovered:,}")
    print(f"  New recovered: {new_recovered:,}")
    print(f"  Additional GT pairs recovered: {new_recovered - v6_recovered:,}")
    print(f"  Avg candidates/S1: {avg_candidates:.2f}")
    print(f"  Max candidates/S1: {max_candidates}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  V6 blocking recall: {v6_recall:.4%}")
    print(f"  New blocking recall: {new_recall:.4%}")
    print(f"  Improvement: {new_recall - v6_recall:+.4%}")
    print(f"  Additional GT pairs recovered: {new_recovered - v6_recovered:,}")
    print(f"  Avg candidates/S1: {avg_candidates:.2f}")
    print(f"  Max candidates/S1: {max_candidates}")
    print(f"  Runtime: {time.time()-t0:.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
