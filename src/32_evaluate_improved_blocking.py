"""
32_evaluate_improved_blocking.py

Evaluate improved blocking recall against ground truth.
Compare V6 vs improved blocking on the same 100K training S1 sample.
"""

from pathlib import Path
import csv
import time
from collections import defaultdict

import pandas as pd

ROOT = Path(r"C:\Users\adity\hackathon")
RAW_TRAIN = ROOT / "data" / "raw" / "train" / "train"
PROCESSED = ROOT / "data" / "processed"

S1_FILE = RAW_TRAIN / "train_source1.tsv"
GT_FILE = RAW_TRAIN / "train_ground_truth.tsv"

N_S1_SAMPLES = 100_000


def load_gt(s1_ids):
    """Load ground truth for given S1 IDs."""
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
    return gt_lookup


def load_blocking_v6(s1_ids):
    """Load V6 blocking candidates."""
    s2_cand = pd.read_csv(PROCESSED / "blocking_pilot_s1_s2_v4.csv", sep=",", dtype=str,
                         keep_default_na=False, na_filter=False)
    s3_cand = pd.read_csv(PROCESSED / "blocking_pilot_s1_s3_v4.csv", sep=",", dtype=str,
                         keep_default_na=False, na_filter=False)
    s2_cand = s2_cand[s2_cand["source1_entity_id"].isin(s1_ids)]
    s3_cand = s3_cand[s3_cand["source1_entity_id"].isin(s1_ids)]

    cand_dict = {}
    for row in s2_cand.itertuples(index=False):
        s1_id = str(row.source1_entity_id)
        raw = str(row.candidate_entity_ids)
        if raw:
            cands = {x.strip() for x in raw.split("|") if x.strip()}
        else:
            cands = set()
        cand_dict.setdefault(s1_id, set()).update(cands)

    for row in s3_cand.itertuples(index=False):
        s1_id = str(row.source1_entity_id)
        raw = str(row.candidate_entity_ids)
        if raw:
            cands = {x.strip() for x in raw.split("|") if x.strip()}
        else:
            cands = set()
        cand_dict.setdefault(s1_id, set()).update(cands)

    return cand_dict


def load_blocking_improved(s1_ids, source_name):
    """Load improved blocking candidates."""
    file_path = PROCESSED / f"improved_blocking_{source_name}.tsv"
    if not file_path.exists():
        return {}

    cand_dict = defaultdict(set)
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                s1_id = row[0]
                s2_id = row[1]
                if s1_id in s1_ids:
                    cand_dict[s1_id].add(s2_id)

    return dict(cand_dict)


def evaluate_blocking(cand_dict, gt_lookup, label):
    """Evaluate blocking recall."""
    total_gt_pairs = 0
    recovered_pairs = 0
    entities_with_gt = 0
    entities_with_recovery = 0
    total_candidates = 0
    max_candidates = 0
    zero_candidates = 0

    for s1_id, gt_matches in gt_lookup.items():
        cands = cand_dict.get(s1_id, set())
        total_candidates += len(cands)
        max_candidates = max(max_candidates, len(cands))
        if not cands:
            zero_candidates += 1

        if not gt_matches:
            continue

        entities_with_gt += 1
        total_gt_pairs += len(gt_matches)

        recovered = cands & gt_matches
        recovered_pairs += len(recovered)
        if recovered:
            entities_with_recovery += 1

    pair_recall = recovered_pairs / total_gt_pairs if total_gt_pairs else 0
    entity_recall = entities_with_recovery / entities_with_gt if entities_with_gt else 0
    avg_candidates = total_candidates / len(cand_dict) if cand_dict else 0

    print(f"\n  {label}:")
    print(f"    Pair-level recall: {pair_recall:.4%}")
    print(f"    Entity-level recall: {entity_recall:.4%}")
    print(f"    Total GT pairs: {total_gt_pairs:,}")
    print(f"    Recovered pairs: {recovered_pairs:,}")
    print(f"    Missed pairs: {total_gt_pairs - recovered_pairs:,}")
    print(f"    Avg candidates/S1: {avg_candidates:.2f}")
    print(f"    Max candidates/S1: {max_candidates}")
    print(f"    Zero-candidate S1: {zero_candidates:,}")

    return {
        "pair_recall": pair_recall,
        "entity_recall": entity_recall,
        "total_gt_pairs": total_gt_pairs,
        "recovered_pairs": recovered_pairs,
        "avg_candidates": avg_candidates,
        "max_candidates": max_candidates,
    }


def main():
    print("=" * 78)
    print("EVALUATE IMPROVED BLOCKING")
    print("=" * 78)

    # Load S1 sample
    print("\n[1] Loading S1 sample...")
    s1 = pd.read_csv(S1_FILE, sep="\t", dtype=str, nrows=N_S1_SAMPLES,
                     keep_default_na=False, na_filter=False)
    s1_ids = set(s1["entity_id"].astype(str))
    print(f"  S1 entities: {len(s1_ids):,}")

    # Load ground truth
    print("\n[2] Loading ground truth...")
    gt_lookup = load_gt(s1_ids)
    total_gt_pairs = sum(len(v) for v in gt_lookup.values())
    print(f"  GT pairs: {total_gt_pairs:,}")

    # Load V6 blocking
    print("\n[3] Loading V6 blocking candidates...")
    v6_cand = load_blocking_v6(s1_ids)
    print(f"  V6 candidates loaded: {len(v6_cand):,} S1")

    # Load improved blocking
    print("\n[4] Loading improved blocking candidates...")
    imp_s2 = load_blocking_improved(s1_ids, "s2")
    imp_s3 = load_blocking_improved(s1_ids, "s3")
    print(f"  Improved S2 candidates: {len(imp_s2):,} S1")
    print(f"  Improved S3 candidates: {len(imp_s3):,} S1")

    # Combine improved S2 + S3
    imp_cand = {}
    for s1_id in s1_ids:
        cands = set()
        cands.update(imp_s2.get(s1_id, set()))
        cands.update(imp_s3.get(s1_id, set()))
        imp_cand[s1_id] = cands

    # Evaluate
    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)

    v6_metrics = evaluate_blocking(v6_cand, gt_lookup, "V6 Blocking")
    imp_metrics = evaluate_blocking(imp_cand, gt_lookup, "Improved Blocking")

    # Comparison
    print("\n" + "=" * 78)
    print("COMPARISON")
    print("=" * 78)
    print(f"  {'Metric':<30} {'V6':>12} {'Improved':>12} {'Delta':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Pair recall':<30} {v6_metrics['pair_recall']:>11.4%} {imp_metrics['pair_recall']:>11.4%} {imp_metrics['pair_recall']-v6_metrics['pair_recall']:>+11.4%}")
    print(f"  {'Entity recall':<30} {v6_metrics['entity_recall']:>11.4%} {imp_metrics['entity_recall']:>11.4%} {imp_metrics['entity_recall']-v6_metrics['entity_recall']:>+11.4%}")
    print(f"  {'Avg candidates/S1':<30} {v6_metrics['avg_candidates']:>12.2f} {imp_metrics['avg_candidates']:>12.2f} {imp_metrics['avg_candidates']-v6_metrics['avg_candidates']:>+12.2f}")
    print(f"  {'Max candidates/S1':<30} {v6_metrics['max_candidates']:>12} {imp_metrics['max_candidates']:>12} {imp_metrics['max_candidates']-v6_metrics['max_candidates']:>+12}")

    # Theoretical max F0.5
    print("\n" + "=" * 78)
    print("THEORETICAL MAX F0.5")
    print("=" * 78)
    blocking_recall = imp_metrics["pair_recall"]
    # If precision = 1.0, F0.5 = 1.25 * 1.0 * r / (0.25 * 1.0 + r)
    f05_max = 1.25 * 1.0 * blocking_recall / (0.25 * 1.0 + blocking_recall)
    print(f"  Blocking recall: {blocking_recall:.4%}")
    print(f"  Theoretical max F0.5 (precision=1.0): {f05_max:.4f}")
    print(f"  Target F0.5: 0.98")
    print(f"  {'PASS' if f05_max >= 0.98 else 'FAIL'}: {'Can reach 0.98' if f05_max >= 0.98 else 'Cannot reach 0.98'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
