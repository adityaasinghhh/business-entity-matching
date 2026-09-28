from pathlib import Path
from collections import Counter
import pandas as pd
import re
import time

ROOT = Path(r"C:\Users\adity\hackathon")

S1_FILE = ROOT / "data/raw/train/train/train_source1.tsv"
S2_FILE = ROOT / "data/raw/train/train/train_source2.tsv"
GT_FILE = ROOT / "data/raw/train/train/train_ground_truth.tsv"

S1_LIMIT = 100_000
S2_CHUNK = 100_000


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


def prepare(df):
    df["name_norm"] = df["business_name"].map(norm)
    df["addr_norm"] = df["business_address"].map(norm)

    df["tokens"] = df["name_norm"].map(toks)
    df["t1"] = df["tokens"].map(lambda x: x[0] if x else "")
    df["t2"] = df["tokens"].map(lambda x: x[1] if len(x) > 1 else "")

    df["p3"] = df["name_norm"].map(lambda x: pref(x, 3))
    df["p5"] = df["name_norm"].map(lambda x: pref(x, 5))

    df["num"] = df["addr_norm"].map(anum)

    df["addr_tokens"] = df["addr_norm"].map(
        lambda x: {t for t in x.split() if len(t) >= 3}
    )

    return df


print("=" * 78)
print("MISSED GROUND-TRUTH PAIR DIAGNOSTIC")
print("=" * 78)

# ------------------------------------------------------------
# Load S1 sample
# ------------------------------------------------------------

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    nrows=S1_LIMIT,
    keep_default_na=False,
    na_filter=False,
)

s1 = prepare(s1)

s1_ids = set(s1["entity_id"].astype(str))

print(f"S1 sample: {len(s1):,}")

# ------------------------------------------------------------
# Load ground truth
# ------------------------------------------------------------

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str,
    usecols=["source1_entity_id", "matched_entity_ids"],
    keep_default_na=False,
    na_filter=False,
)

gt = gt[
    gt["source1_entity_id"].astype(str).isin(s1_ids)
].copy()

gt_map = {}

target_ids = set()

for r in gt.itertuples(index=False):
    sid = str(r.source1_entity_id)
    raw = str(r.matched_entity_ids).strip()

    matches = {
        x.strip()
        for x in raw.split(",")
        if x.strip()
    }

    gt_map[sid] = matches
    target_ids.update(matches)

print(f"GT S1 records: {len(gt_map):,}")
print(f"Unique true S2 IDs: {len(target_ids):,}")

# ------------------------------------------------------------
# Build S1 lookup
# ------------------------------------------------------------

s1_lookup = {
    str(r.entity_id): r
    for r in s1.itertuples(index=False)
}

# ------------------------------------------------------------
# Stream S2 and keep ONLY ground-truth target records
# ------------------------------------------------------------

print("\nScanning S2 for ground-truth records...")

start = time.time()

s2_parts = []
found_ids = set()

for chunk in pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    chunksize=S2_CHUNK,
    keep_default_na=False,
    na_filter=False,
):
    chunk["entity_id"] = chunk["entity_id"].astype(str)

    mask = chunk["entity_id"].isin(target_ids)

    if mask.any():
        part = chunk.loc[
            mask,
            [
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ],
        ].copy()

        s2_parts.append(part)
        found_ids.update(part["entity_id"].tolist())

elapsed = time.time() - start

print(
    f"S2 scan complete in {elapsed / 60:.2f} minutes."
)

print(f"True S2 IDs found: {len(found_ids):,}")

missing_target_ids = target_ids - found_ids

if missing_target_ids:
    print(
        f"WARNING: {len(missing_target_ids):,} GT IDs "
        f"were not found in S2."
    )

if not s2_parts:
    raise RuntimeError("No ground-truth S2 records were found.")

s2 = pd.concat(s2_parts, ignore_index=True)
s2 = prepare(s2)

s2_lookup = {
    str(r.entity_id): r
    for r in s2.itertuples(index=False)
}

print(f"S2 GT records loaded: {len(s2_lookup):,}")

# ------------------------------------------------------------
# Evaluate each TRUE pair against every blocking rule
# ------------------------------------------------------------

rule_names = [
    "exact",
    "token12",
    "token_number",
    "prefix5",
    "token_address",
]

rule_hits = Counter()
union_hits = 0
total_pairs = 0

country_mismatch = 0
name_exact_match = 0
same_t1 = 0
same_t2 = 0
same_p3 = 0
same_p5 = 0
same_number = 0
shared_address_token = 0

missed_examples = []

for sid, truth_ids in gt_map.items():

    s1r = s1_lookup.get(sid)

    if s1r is None:
        continue

    for cid in truth_ids:

        s2r = s2_lookup.get(cid)

        if s2r is None:
            continue

        total_pairs += 1

        same_country = (
            str(s1r.country).strip().lower()
            == str(s2r.country).strip().lower()
        )

        exact = (
            bool(s1r.name_norm)
            and s1r.name_norm == s2r.name_norm
            and same_country
        )

        token12 = (
            bool(s1r.t1)
            and bool(s1r.t2)
            and s1r.t1 == s2r.t1
            and s1r.t2 == s2r.t2
            and same_country
        )

        token_number = (
            bool(s1r.t1)
            and bool(s1r.num)
            and s1r.t1 == s2r.t1
            and s1r.num == s2r.num
            and same_country
        )

        prefix5 = (
            bool(s1r.p5)
            and s1r.p5 == s2r.p5
            and same_country
        )

        token_address = (
            bool(s1r.t1)
            and s1r.t1 == s2r.t1
            and bool(
                s1r.addr_tokens
                & s2r.addr_tokens
            )
            and same_country
        )

        hits = {
            "exact": exact,
            "token12": token12,
            "token_number": token_number,
            "prefix5": prefix5,
            "token_address": token_address,
        }

        for rule, hit in hits.items():
            if hit:
                rule_hits[rule] += 1

        if any(hits.values()):
            union_hits += 1
        else:
            if len(missed_examples) < 100:
                missed_examples.append({
                    "source1_entity_id": sid,
                    "source2_entity_id": cid,
                    "s1_name": s1r.name_norm,
                    "s2_name": s2r.name_norm,
                    "s1_address": s1r.addr_norm,
                    "s2_address": s2r.addr_norm,
                    "s1_country": s1r.country,
                    "s2_country": s2r.country,
                })

        if not same_country:
            country_mismatch += 1

        if s1r.name_norm and s1r.name_norm == s2r.name_norm:
            name_exact_match += 1

        if s1r.t1 and s1r.t1 == s2r.t1:
            same_t1 += 1

        if s1r.t2 and s1r.t2 == s2r.t2:
            same_t2 += 1

        if s1r.p3 and s1r.p3 == s2r.p3:
            same_p3 += 1

        if s1r.p5 and s1r.p5 == s2r.p5:
            same_p5 += 1

        if s1r.num and s1r.num == s2r.num:
            same_number += 1

        if s1r.addr_tokens & s2r.addr_tokens:
            shared_address_token += 1

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

if total_pairs == 0:
    raise RuntimeError("No valid GT pairs were evaluated.")

print("\n" + "=" * 78)
print("GROUND-TRUTH BLOCKING COVERAGE")
print("=" * 78)

print(f"Total GT pairs evaluated : {total_pairs:,}")

for rule in rule_names:
    count = rule_hits[rule]
    print(
        f"{rule:18s}: "
        f"{count:,} "
        f"({count / total_pairs:.4%})"
    )

print(
    f"{'ANY CURRENT RULE':18s}: "
    f"{union_hits:,} "
    f"({union_hits / total_pairs:.4%})"
)

print("\n" + "-" * 78)
print("PAIR CHARACTERISTICS")
print("-" * 78)

characteristics = [
    ("country mismatch", country_mismatch),
    ("exact normalized name", name_exact_match),
    ("same first token", same_t1),
    ("same second token", same_t2),
    ("same prefix3", same_p3),
    ("same prefix5", same_p5),
    ("same address number", same_number),
    ("shared address token", shared_address_token),
]

for name, count in characteristics:
    print(
        f"{name:24s}: "
        f"{count:,} "
        f"({count / total_pairs:.4%})"
    )

missed_count = total_pairs - union_hits

print("\n" + "-" * 78)
print(f"TRUE PAIRS MISSED BY ALL CURRENT RULES: {missed_count:,}")
print(
    f"MISSED RATE: "
    f"{missed_count / total_pairs:.4%}"
)

# ------------------------------------------------------------
# Save only a tiny sample of missed pairs
# ------------------------------------------------------------

out = ROOT / "data/processed/experiments"
out.mkdir(parents=True, exist_ok=True)

sample_file = out / "missed_ground_truth_sample_100k.csv"

pd.DataFrame(missed_examples).to_csv(
    sample_file,
    index=False,
)

print(f"\nSaved missed-pair sample: {sample_file}")

print("\n" + "=" * 78)
print("DIAGNOSTIC COMPLETE")
print("=" * 78)
