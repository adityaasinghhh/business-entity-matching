from pathlib import Path
import re
import numpy as np
import pandas as pd
import xgboost as xgb
from rapidfuzz import fuzz

ROOT = Path(r"C:\Users\adity\hackathon")

S1_FILE = ROOT / r"data\raw\train\train\test\test\test_source1.tsv"
S2_FILE = ROOT / r"data\raw\train\train\test\test\test_source2.tsv"
CAND_FILE = ROOT / r"data\processed\blocking_test_s1_s2_v6.tsv"
MODEL_FILE = ROOT / r"models\xgboost_entity_match_v3.json"

N = 5000

FEATURES = [
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_exact",
    "name_length_diff",
    "address_ratio",
    "address_partial_ratio",
    "address_token_set_ratio",
    "address_exact",
    "address_length_diff",
    "country_match",
    "shared_token_count",
    "shared_token_ratio",
    "address_number_match",
    "address_number_count",
    "candidate_score",
    "candidate_rank",
]

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
SPACE = re.compile(r"\s+")
NUM = re.compile(r"\d+")


def norm(x):
    x = "" if x is None else str(x).lower().strip()
    x = URL_RE.sub(" ", x)
    x = NON_WORD.sub(" ", x)
    return SPACE.sub(" ", x).strip()


def toks(x):
    return set(x.split()) if x else set()


def nums(x):
    return set(NUM.findall(x)) if x else set()


def features(a, b, rank):

    n1, n2 = norm(a["business_name"]), norm(b["business_name"])
    ad1, ad2 = norm(a["business_address"]), norm(b["business_address"])
    c1, c2 = norm(a["country"]), norm(b["country"])

    t1, t2 = toks(n1), toks(n2)
    num1, num2 = nums(ad1), nums(ad2)

    nr = fuzz.ratio(n1, n2)
    npr = fuzz.partial_ratio(n1, n2)
    nts = fuzz.token_sort_ratio(n1, n2)
    nset = fuzz.token_set_ratio(n1, n2)

    ar = fuzz.ratio(ad1, ad2)
    apr = fuzz.partial_ratio(ad1, ad2)
    aset = fuzz.token_set_ratio(ad1, ad2)

    shared = t1 & t2
    shared_nums = num1 & num2

    country = int(bool(c1) and c1 == c2)
    ne = int(bool(n1) and n1 == n2)
    ae = int(bool(ad1) and ad1 == ad2)
    anm = int(bool(num1) and bool(shared_nums))

    score = (
        .45 * nset
        + .20 * nr
        + .15 * npr
        + .10 * aset
        + .05 * ar
        + 5 * country
        + 3 * anm
    )

    return [
        nr,
        npr,
        nts,
        nset,
        ne,
        abs(len(n1) - len(n2)),
        ar,
        apr,
        aset,
        ae,
        abs(len(ad1) - len(ad2)),
        country,
        len(shared),
        len(shared) / len(t1) if t1 else 0,
        anm,
        len(shared_nums),
        score,
        rank,
    ]


print("=" * 70)
print("CORRECT V6 MODEL DIAGNOSTIC")
print("=" * 70)

model = xgb.XGBClassifier()
model.load_model(str(MODEL_FILE))

if model.get_booster().feature_names:
    if list(model.get_booster().feature_names) != FEATURES:
        raise RuntimeError("FEATURE ORDER MISMATCH")

print("Model OK.")

# ------------------------------------------------------------
# Read actual V5 candidates
# ------------------------------------------------------------

cand = pd.read_csv(
    CAND_FILE,
    sep="\t",
    dtype=str,
    nrows=N,
    keep_default_na=False,
    na_filter=False,
)

print(f"Candidate rows: {len(cand):,}")

# ------------------------------------------------------------
# Read S1 records and index by ID
# ------------------------------------------------------------

needed_s1 = set(cand["source1_entity_id"])

s1_lookup = {}

for chunk in pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    chunksize=200_000,
    keep_default_na=False,
    na_filter=False,
):
    found = chunk[
        chunk["entity_id"].isin(needed_s1)
    ]

    for r in found.itertuples(index=False):
        s1_lookup[str(r.entity_id)] = {
            "business_name": r.business_name,
            "business_address": r.business_address,
            "country": r.country,
        }

    if len(s1_lookup) == len(needed_s1):
        break

print(f"S1 records loaded: {len(s1_lookup):,}")

# ------------------------------------------------------------
# Collect S2 IDs
# ------------------------------------------------------------

needed_s2 = set()

for value in cand["candidate_entity_ids"]:

    if not value:
        continue

    for cid in value.split(","):
        cid = cid.strip()
        if cid:
            needed_s2.add(cid)

print(f"S2 candidates needed: {len(needed_s2):,}")

# ------------------------------------------------------------
# Load only required S2
# ------------------------------------------------------------

s2_lookup = {}

for chunk in pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    chunksize=200_000,
    keep_default_na=False,
    na_filter=False,
):

    found = chunk[
        chunk["entity_id"].isin(needed_s2)
    ]

    for r in found.itertuples(index=False):
        s2_lookup[str(r.entity_id)] = {
            "business_name": r.business_name,
            "business_address": r.business_address,
            "country": r.country,
        }

    if len(s2_lookup) == len(needed_s2):
        break

print(f"S2 records loaded: {len(s2_lookup):,}")

# ------------------------------------------------------------
# Score
# ------------------------------------------------------------

results = []

for i, row in cand.iterrows():

    sid = str(row["source1_entity_id"])

    s1 = s1_lookup.get(sid)

    if s1 is None:
        continue

    ids = [
        x.strip()
        for x in str(row["candidate_entity_ids"]).split(",")
        if x.strip()
    ]

    if not ids:
        continue

    X = []
    valid_ids = []

    for rank, cid in enumerate(ids, 1):

        s2 = s2_lookup.get(cid)

        if s2 is None:
            continue

        X.append(
            features(
                s1,
                s2,
                rank
            )
        )

        valid_ids.append(cid)

    if not X:
        continue

    probs = model.predict_proba(
        np.asarray(X, dtype=np.float32)
    )[:, 1]

    order = np.argsort(probs)[::-1]

    best = float(probs[order[0]])

    second = (
        float(probs[order[1]])
        if len(order) > 1
        else 0.0
    )

    results.append({
        "s1_id": sid,
        "candidate_count": len(valid_ids),
        "best_probability": best,
        "second_probability": second,
        "margin": best - second,
    })

    if (i + 1) % 500 == 0:
        print(
            f"\rScored {i + 1:,}/{N:,}",
            end="",
            flush=True,
        )

print("\n")

d = pd.DataFrame(results)

out = ROOT / "outputs" / "model_diagnostic_v5.csv"
d.to_csv(out, index=False)

print("=" * 70)
print("RESULT")
print("=" * 70)

print(f"Scored: {len(d):,}")

print("\nBest probability:")
print(
    d["best_probability"].describe(
        percentiles=[
            .10, .25, .50, .75,
            .90, .95, .99
        ]
    )
)

print("\nTHRESHOLD COUNTS")

for t in [.30, .40, .50, .60, .70, .80, .90, .95, .99]:

    n = (d["best_probability"] >= t).sum()

    print(
        f"{t:.2f}: "
        f"{n:,}/{len(d):,} "
        f"({n / len(d) * 100:.2f}%)"
    )

print("\nMARGIN:")
print(d["margin"].describe())

print(f"\nSaved: {out}")