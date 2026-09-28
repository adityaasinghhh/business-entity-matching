"""
08_evaluate_v8.py

Evaluate V8 blocking recall against ground truth.
Compares V8 output with V6 output on the same S1 subset.

Does NOT modify any existing files.
Creates a new evaluation report.

Usage:
  python 08_evaluate_v8.py --limit 10000
  python 08_evaluate_v8.py --limit 100000
"""

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\adity\hackathon")
RAW_TRAIN = ROOT / "data" / "raw" / "train" / "train"
PROCESSED = ROOT / "data" / "processed"

GT_FILE = RAW_TRAIN / "train_ground_truth.tsv"
V6_BLOCKING = PROCESSED / "blocking_pilot_s1_s2_v4.csv"
V8_S2_BLOCKING = PROCESSED / "v8_output" / "blocking_v8_s2.tsv"


def load_ground_truth(s1_ids, source_prefix="S2-"):
    """Load ground truth filtered to specific S1 IDs and S2 matches."""
    gt = pd.read_csv(GT_FILE, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    gt = gt[gt["source1_entity_id"].isin(s1_ids)]

    gt_lookup = {}
    for row in gt.itertuples(index=False):
        s1_id = row.source1_entity_id
        raw = row.matched_entity_ids.strip()
        if not raw:
            gt_lookup[s1_id] = set()
            continue
        # Filter to only S2 or S3 matches
        matches = {
            x.strip() for x in raw.split(",")
            if x.strip().startswith(source_prefix)
        }
        gt_lookup[s1_id] = matches

    return gt_lookup


def load_blocking_output(file_path, s1_ids=None):
    """
    Load blocking output and return dict: {s1_id: {s2_id: set(rules)}}
    """
    if not Path(file_path).exists():
        print(f"  [WARN] Blocking file not found: {file_path}")
        return {}

    df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)

    # Handle different column names
    if "source1_entity_id" in df.columns:
        s1_col = "source1_entity_id"
    elif "s1_id" in df.columns:
        s1_col = "s1_id"
    else:
        s1_col = df.columns[0]

    if "source_entity_id" in df.columns:
        s2_col = "source_entity_id"
    elif "candidate_entity_ids" in df.columns:
        # V6 format: pipe-separated or comma-separated IDs in one row
        s2_col = "candidate_entity_ids"
    else:
        s2_col = df.columns[1]

    candidates = defaultdict(lambda: defaultdict(set))

    if s2_col == "candidate_entity_ids":
        # V6 format: one row per S1, IDs separated by | or ,
        for row in df.itertuples(index=False):
            s1_id = str(getattr(row, s1_col))
            if s1_ids and s1_id not in s1_ids:
                continue
            raw_ids = str(getattr(row, s2_col))
            if not raw_ids:
                continue
            # Try pipe first, then comma
            if "|" in raw_ids:
                ids = [x.strip() for x in raw_ids.split("|") if x.strip()]
            else:
                ids = [x.strip() for x in raw_ids.split(",") if x.strip()]
            for cid in ids:
                candidates[s1_id][cid].add("v6_blocking")
    else:
        # V8 format: one row per candidate pair
        for row in df.itertuples(index=False):
            s1_id = str(getattr(row, s1_col))
            if s1_ids and s1_id not in s1_ids:
                continue
            s2_id = str(getattr(row, s2_col))
            rules_col = None
            for col in df.columns:
                if "rule" in col.lower():
                    rules_col = col
                    break
            if rules_col:
                rules = str(getattr(row, rules_col))
                rule_set = {r.strip() for r in rules.split(",") if r.strip()}
            else:
                rule_set = {"blocking"}
            candidates[s1_id][s2_id].update(rule_set)

    return candidates


def evaluate_blocking(candidates, gt_lookup):
    """Compute blocking recall metrics."""
    total_gt_pairs = 0
    recovered_pairs = 0
    entities_with_gt = 0
    entities_with_recovery = 0
    entities_with_all_matches = 0
    zero_candidate_entities = 0
    total_candidates = 0
    max_candidates = 0
    rule_contributions = defaultdict(int)

    for s1_id, gt_matches in gt_lookup.items():
        cand_dict = candidates.get(s1_id, {})
        cand_count = len(cand_dict)
        total_candidates += cand_count
        max_candidates = max(max_candidates, cand_count)

        if cand_count == 0:
            zero_candidate_entities += 1

        if gt_matches:
            entities_with_gt += 1
            total_gt_pairs += len(gt_matches)

            recovered = set(cand_dict.keys()) & gt_matches
            recovered_pairs += len(recovered)

            if recovered:
                entities_with_recovery += 1

            if recovered == gt_matches:
                entities_with_all_matches += 1

        # Track rule contributions
        for s2_id, rules in cand_dict.items():
            if s2_id in gt_matches:
                for rule in rules:
                    rule_contributions[rule] += 1

    pair_recall = recovered_pairs / total_gt_pairs if total_gt_pairs else 0
    entity_recall = entities_with_recovery / entities_with_gt if entities_with_gt else 0
    all_match_rate = entities_with_all_matches / entities_with_gt if entities_with_gt else 0
    avg_candidates = total_candidates / len(gt_lookup) if gt_lookup else 0

    return {
        "total_gt_pairs": total_gt_pairs,
        "recovered_pairs": recovered_pairs,
        "entities_with_gt": entities_with_gt,
        "entities_with_recovery": entities_with_recovery,
        "entities_with_all_matches": entities_with_all_matches,
        "zero_candidate_entities": zero_candidate_entities,
        "total_candidates": total_candidates,
        "max_candidates": max_candidates,
        "avg_candidates": avg_candidates,
        "pair_recall": pair_recall,
        "entity_recall": entity_recall,
        "all_match_rate": all_match_rate,
        "rule_contributions": dict(rule_contributions),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate V8 blocking")
    parser.add_argument("--limit", type=int, default=10000, help="Number of S1 entities")
    args = parser.parse_args()

    print("=" * 72)
    print(f"V8 BLOCKING EVALUATION -- {args.limit:,} S1 entities")
    print("=" * 72)

    # Load S1 IDs
    s1_file = ROOT / "data" / "raw" / "train" / "train" / "train_source1.tsv"
    s1_df = pd.read_csv(s1_file, sep="\t", dtype=str, nrows=args.limit,
                        keep_default_na=False, na_filter=False)
    s1_ids = set(s1_df["entity_id"].astype(str))
    print(f"\nS1 entities: {len(s1_ids):,}")

    # Load ground truth
    print("Loading ground truth...")
    gt_lookup = load_ground_truth(s1_ids, "S2-")
    total_gt = sum(len(v) for v in gt_lookup.values())
    entities_with_gt = sum(1 for v in gt_lookup.values() if v)
    print(f"  GT pairs (S2): {total_gt:,}")
    print(f"  S1 with S2 GT: {entities_with_gt:,}")

    # Load V8 blocking
    print("\nLoading V8 blocking output...")
    v8_candidates = load_blocking_output(V8_S2_BLOCKING, s1_ids)
    print(f"  V8 candidates loaded: {len(v8_candidates):,} S1 entities")

    # Load V6 blocking
    print("\nLoading V6 blocking output...")
    v6_candidates = load_blocking_output(V6_BLOCKING, s1_ids)
    print(f"  V6 candidates loaded: {len(v6_candidates):,} S1 entities")

    # Evaluate V8
    print(f"\n{'='*72}")
    print("V8 BLOCKING RESULTS")
    print(f"{'='*72}")
    v8_metrics = evaluate_blocking(v8_candidates, gt_lookup)
    print_metrics(v8_metrics)

    # Evaluate V6
    print(f"\n{'='*72}")
    print("V6 BLOCKING RESULTS (baseline)")
    print(f"{'='*72}")
    v6_metrics = evaluate_blocking(v6_candidates, gt_lookup)
    print_metrics(v6_metrics)

    # Comparison
    print(f"\n{'='*72}")
    print("V8 vs V6 COMPARISON")
    print(f"{'='*72}")
    print(f"{'Metric':<35} {'V6':>12} {'V8':>12} {'Delta':>12}")
    print("-" * 72)
    print(f"{'Pair-level recall':<35} {v6_metrics['pair_recall']:>11.4%} {v8_metrics['pair_recall']:>11.4%} {v8_metrics['pair_recall']-v6_metrics['pair_recall']:>+11.4%}")
    print(f"{'Entity-level recall':<35} {v6_metrics['entity_recall']:>11.4%} {v8_metrics['entity_recall']:>11.4%} {v8_metrics['entity_recall']-v6_metrics['entity_recall']:>+11.4%}")
    print(f"{'All-match rate':<35} {v6_metrics['all_match_rate']:>11.4%} {v8_metrics['all_match_rate']:>11.4%} {v8_metrics['all_match_rate']-v6_metrics['all_match_rate']:>+11.4%}")
    print(f"{'Avg candidates/S1':<35} {v6_metrics['avg_candidates']:>12.2f} {v8_metrics['avg_candidates']:>12.2f} {v8_metrics['avg_candidates']-v6_metrics['avg_candidates']:>+12.2f}")
    print(f"{'Max candidates/S1':<35} {v6_metrics['max_candidates']:>12} {v8_metrics['max_candidates']:>12} {v8_metrics['max_candidates']-v6_metrics['max_candidates']:>+12}")
    print(f"{'Zero-candidate S1':<35} {v6_metrics['zero_candidate_entities']:>12,} {v8_metrics['zero_candidate_entities']:>12,} {v8_metrics['zero_candidate_entities']-v6_metrics['zero_candidate_entities']:>+12,}")
    print(f"{'Total candidates':<35} {v6_metrics['total_candidates']:>12,} {v8_metrics['total_candidates']:>12,} {v8_metrics['total_candidates']-v6_metrics['total_candidates']:>+12,}")

    # Rule contributions
    if v8_metrics["rule_contributions"]:
        print(f"\n{'='*72}")
        print("V8 RULE CONTRIBUTIONS (GT pairs recovered by each rule)")
        print(f"{'='*72}")
        for rule, count in sorted(v8_metrics["rule_contributions"].items(), key=lambda x: -x[1]):
            pct = count / v8_metrics["total_gt_pairs"] if v8_metrics["total_gt_pairs"] else 0
            print(f"  {rule:25s}: {count:>6,} ({pct:.4%})")


def print_metrics(m):
    print(f"  Pair-level recall:        {m['pair_recall']:.4%}")
    print(f"  Entity-level recall:      {m['entity_recall']:.4%}")
    print(f"  All-match rate:           {m['all_match_rate']:.4%}")
    print(f"  Avg candidates/S1:        {m['avg_candidates']:.2f}")
    print(f"  Max candidates/S1:        {m['max_candidates']}")
    print(f"  Zero-candidate S1:        {m['zero_candidate_entities']:,}")
    print(f"  Total candidates:         {m['total_candidates']:,}")
    print(f"  GT pairs recovered:       {m['recovered_pairs']:,} / {m['total_gt_pairs']:,}")


if __name__ == "__main__":
    main()
