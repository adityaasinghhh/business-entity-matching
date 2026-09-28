from pathlib import Path
from collections import defaultdict
import pandas as pd
import re
import time

ROOT = Path(r"C:\Users\adity\hackathon")

S1_FILE = ROOT / "data/raw/train/train/train_source1.tsv"
S2_FILE = ROOT / "data/raw/train/train/train_source2.tsv"
GT_FILE = ROOT / "data/raw/train/train/train_ground_truth.tsv"

S1_LIMIT = 100_000
S2_CHUNK = 100_000
MAX_BLOCK = 500
TOP_K = 50

def norm(x):
    x = "" if pd.isna(x) else str(x).lower().strip()
    x = re.sub(r"https?://\S+|www\.\S+", " ", x)
    x = re.sub(r"[^\w\s]", " ", x, flags=re.UNICODE)
    return re.sub(r"\s+", " ", x).strip()

def toks(x):
    return [t for t in x.split() if len(t) >= 2]

def pref(x, n):
    z = re.sub(r"\s+", "", x)
    return z[:n] if len(z) >= n else ""

def anum(x):
    m = re.search(r"\b\d{1,6}\b", x)
    return m.group(0) if m else ""

print("=" * 72)
print("FAST BLOCKING RECALL BENCHMARK")
print("=" * 72)

s1 = pd.read_csv(
    S1_FILE, sep="\t", dtype=str, nrows=S1_LIMIT,
    keep_default_na=False, na_filter=False
)

s1["name"] = s1["business_name"].map(norm)
s1["addr"] = s1["business_address"].map(norm)
s1["tok"] = s1["name"].map(toks)
s1["t1"] = s1["tok"].map(lambda x: x[0] if x else "")
s1["t2"] = s1["tok"].map(lambda x: x[1] if len(x) > 1 else "")
s1["p3"] = s1["name"].map(lambda x: pref(x, 3))
s1["p5"] = s1["name"].map(lambda x: pref(x, 5))
s1["num"] = s1["addr"].map(anum)
s1["at"] = s1["addr"].map(lambda x: [t for t in x.split() if len(t) >= 3][:2])

print(f"S1: {len(s1):,}")

# ------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str,
    usecols=["source1_entity_id", "matched_entity_ids"],
    keep_default_na=False,
    na_filter=False
)

s1_ids = set(s1["entity_id"].astype(str))
gt = gt[gt["source1_entity_id"].astype(str).isin(s1_ids)]

gt_map = {}

for r in gt.itertuples(index=False):
    raw = str(r.matched_entity_ids).strip()
    gt_map[str(r.source1_entity_id)] = (
        {x.strip() for x in raw.split(",") if x.strip()}
        if raw else set()
    )

print(f"GT loaded: {len(gt_map):,}")

# ------------------------------------------------------------
# Four independent experiments.
# ------------------------------------------------------------

configs = {
    "BASELINE_V6": [
        "exact",
        "token12",
        "token_number",
        "prefix5",
    ],
    "PLUS_TOKEN2": [
        "exact",
        "token12",
        "token_number",
        "prefix5",
        "token2",
    ],
    "PLUS_PREFIX3": [
        "exact",
        "token12",
        "token_number",
        "prefix5",
        "prefix3",
    ],
    "PLUS_ADDRESS_NUMBER": [
        "exact",
        "token12",
        "token_number",
        "prefix5",
        "address_number",
    ],
    "PLUS_TOKEN_ADDRESS": [
        "exact",
        "token12",
        "token_number",
        "prefix5",
        "token_address",
    ],
}

weights = {
    "exact": 4,
    "token12": 3,
    "token_number": 3,
    "prefix5": 1,
    "token2": 2,
    "prefix3": 1,
    "address_number": 1,
    "token_address": 2,
}

indexes = {
    name: defaultdict(list)
    for name in [
        "exact",
        "token12",
        "token_number",
        "prefix5",
        "token2",
        "prefix3",
        "address_number",
        "token_address",
    ]
}

# ------------------------------------------------------------
# Build S2 indexes in one streaming pass.
# ------------------------------------------------------------

print("\nIndexing S2...")

start = time.time()

for chunk in pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    chunksize=S2_CHUNK,
    keep_default_na=False,
    na_filter=False
):
    chunk["name"] = chunk["business_name"].map(norm)
    chunk["addr"] = chunk["business_address"].map(norm)
    chunk["tok"] = chunk["name"].map(toks)
    chunk["t1"] = chunk["tok"].map(lambda x: x[0] if x else "")
    chunk["t2"] = chunk["tok"].map(lambda x: x[1] if len(x) > 1 else "")
    chunk["p3"] = chunk["name"].map(lambda x: pref(x, 3))
    chunk["p5"] = chunk["name"].map(lambda x: pref(x, 5))
    chunk["num"] = chunk["addr"].map(anum)
    chunk["at"] = chunk["addr"].map(
        lambda x: [t for t in x.split() if len(t) >= 3][:2]
    )

    for r in chunk.itertuples(index=False):
        eid = str(r.entity_id)
        c = str(r.country)

        entries = [
            ("exact", (r.name, c)) if r.name else None,
            ("token12", (r.t1, r.t2, c))
                if r.t1 and r.t2 else None,
            ("token_number", (r.t1, r.num, c))
                if r.t1 and r.num else None,
            ("prefix5", (r.p5, c)) if r.p5 else None,
            ("token2", (r.t2, c)) if r.t2 else None,
            ("prefix3", (r.p3, c)) if r.p3 else None,
            ("address_number", (r.num, c)) if r.num else None,
        ]

        for item in entries:
            if item is not None:
                rule, key = item
                indexes[rule][key].append(eid)
        if r.t1 and r.at:
            for a in r.at:
                indexes["token_address"][
                    (r.t1, a, c)
                ].append(eid)

elapsed = time.time() - start

print(f"S2 indexed in {elapsed/60:.2f} minutes.")

# ------------------------------------------------------------
# Remove oversized blocks.
# ------------------------------------------------------------

for rule, idx in indexes.items():
    oversized = [
        k for k, v in idx.items()
        if len(v) > MAX_BLOCK
    ]
    for k in oversized:
        del idx[k]

    print(
        f"{rule:16s}: blocks={len(idx):,} "
        f"oversized_removed={len(oversized):,}"
    )

# ------------------------------------------------------------
# Evaluate candidate K values.
# ------------------------------------------------------------

TOP_KS = [50, 100, 150, 200]

# Only compare the two configurations currently supported by
# the benchmark evidence.
test_configs = {
    "BASELINE_V6": [
        "exact",
        "token12",
        "token_number",
        "prefix5",
    ],
    "PLUS_TOKEN_ADDRESS": [
        "exact",
        "token12",
        "token_number",
        "prefix5",
        "token_address",
    ],
}

results = []

for top_k in TOP_KS:

    print("\n" + "=" * 72)
    print(f"TOP-K = {top_k}")
    print("=" * 72)

    for config_name, rules in test_configs.items():

        print("\n" + "-" * 72)
        print(config_name)
        print("-" * 72)

        pair_gt = 0
        pair_hit = 0
        entity_gt = 0
        entity_hit = 0
        entity_all = 0

        total_candidates = 0
        max_candidates = 0
        zero_candidates = 0

        for r in s1.itertuples(index=False):

            sid = str(r.entity_id)
            country = str(r.country)

            scores = {}

            def collect(rule, key):
                if not key:
                    return

                for cid in indexes[rule].get(key, []):
                    scores[cid] = max(
                        scores.get(cid, 0),
                        weights[rule]
                    )

            if "exact" in rules:
                collect("exact", (r.name, country))

            if "token12" in rules:
                collect("token12", (r.t1, r.t2, country))

            if "token_number" in rules:
                collect(
                    "token_number",
                    (r.t1, r.num, country)
                )

            if "prefix5" in rules:
                collect("prefix5", (r.p5, country))

            if "token2" in rules:
                collect("token2", (r.t2, country))

            if "prefix3" in rules:
                collect("prefix3", (r.p3, country))

            if "address_number" in rules:
                collect(
                    "address_number",
                    (r.num, country)
                )

            if "token_address" in rules and r.t1:
                for a in r.at:
                    collect(
                        "token_address",
                        (r.t1, a, country)
                    )

            ranked = sorted(
                scores.items(),
                key=lambda x: (-x[1], x[0])
            )[:top_k]

            pred = {x[0] for x in ranked}

            total_candidates += len(pred)
            max_candidates = max(max_candidates, len(pred))

            if not pred:
                zero_candidates += 1

            truth = gt_map.get(sid, set())

            if truth:
                entity_gt += 1
                pair_gt += len(truth)

                hit = pred & truth

                pair_hit += len(hit)

                if hit:
                    entity_hit += 1

                if hit == truth:
                    entity_all += 1

        pair_recall = pair_hit / pair_gt if pair_gt else 0
        entity_recall = entity_hit / entity_gt if entity_gt else 0
        all_rate = entity_all / entity_gt if entity_gt else 0
        avg = total_candidates / len(s1)

        print(f"Pair recall       : {pair_recall:.4%}")
        print(f"Entity recall     : {entity_recall:.4%}")
        print(f"All-match rate    : {all_rate:.4%}")
        print(f"Avg candidates    : {avg:.2f}")
        print(f"Max candidates    : {max_candidates}")
        print(f"Zero candidates   : {zero_candidates:,}")

        results.append({
            "config": config_name,
            "top_k": top_k,
            "pair_recall": pair_recall,
            "entity_recall": entity_recall,
            "all_match_rate": all_rate,
            "avg_candidates": avg,
            "max_candidates": max_candidates,
            "zero_candidates": zero_candidates,
        })

out = ROOT / "data/processed/experiments"
out.mkdir(parents=True, exist_ok=True)

result_file = out / "blocking_k_comparison_100k.csv"
pd.DataFrame(results).to_csv(result_file, index=False)

print("\n" + "=" * 72)
print("K COMPARISON COMPLETE")
print("=" * 72)

print(
    pd.DataFrame(results).to_string(index=False)
)

print(f"\nSaved: {result_file}")
out = ROOT / "data/processed/experiments"
out.mkdir(parents=True, exist_ok=True)

result_file = out / "blocking_rule_comparison_100k.csv"
pd.DataFrame(results).to_csv(result_file, index=False)

print("\n" + "=" * 72)
print("COMPARISON COMPLETE")
print("=" * 72)

print(
    pd.DataFrame(results).to_string(index=False)
)

print(f"\nSaved: {result_file}")




