"""
21_diagnose_model_rejection.py

Investigate why the model is rejecting true matches.
Check probability distribution, threshold, and multi-match handling.
"""

from pathlib import Path
import gc
import re
import time
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from rapidfuzz import process, fuzz

ROOT = Path(r"C:\Users\adity\hackathon")
RAW_TRAIN = ROOT / "data" / "raw" / "train" / "train"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

S1_FILE = RAW_TRAIN / "train_source1.tsv"
S2_FILE = RAW_TRAIN / "train_source2.tsv"
S3_FILE = RAW_TRAIN / "train_source3.tsv"
GT_FILE = RAW_TRAIN / "train_ground_truth.tsv"

MODEL_FILE = MODELS / "xgboost_entity_match_v3.json"

N_S1_SAMPLES = 10_000
CHUNK_SIZE = 10_000
TOP_K_FOR_MODEL = 20
DUCKDB_MEMORY = "3GB"
DUCKDB_THREADS = 4

FEATURE_COLUMNS = [
    "name_ratio", "name_partial_ratio", "name_token_sort_ratio",
    "name_token_set_ratio", "name_exact", "name_length_diff",
    "address_ratio", "address_partial_ratio", "address_token_set_ratio",
    "address_exact", "address_length_diff", "country_match",
    "shared_token_count", "shared_token_ratio", "address_number_match",
    "address_number_count", "candidate_score", "candidate_rank",
]

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
NUMBER_RE = re.compile(r"\d+")


def normalize_series(s):
    s = s.fillna("").astype(str).astype(object)
    return (s.str.lower().str.strip()
             .str.replace(URL_RE, " ", regex=True)
             .str.replace(NON_WORD_RE, " ", regex=True)
             .str.replace(SPACE_RE, " ", regex=True)
             .str.strip())


def load_model():
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_FILE))
    return model


def build_or_open_cache(con):
    for table, path in (("s2", S2_FILE), ("s3", S3_FILE)):
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name=?",
            [table],
        ).fetchone()[0]
        if exists:
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  Lookup cache: {table} ready ({count:,} rows)")
            continue
        print(f"  Building lookup cache for {table}...")
        t = time.time()
        con.execute(f"""
            CREATE TABLE {table} AS
            SELECT
                CAST(entity_id AS VARCHAR) AS entity_id,
                CAST(business_name AS VARCHAR) AS business_name,
                CAST(business_address AS VARCHAR) AS business_address,
                CAST(country AS VARCHAR) AS country
            FROM read_csv('{str(path).replace(chr(39), chr(39)*2)}', delim='\t', header=true,
                          quote='"', escape='"', nullstr='')
        """)
        con.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_id ON {table}(entity_id)")
        con.execute("CHECKPOINT")
        print(f"    {table}: ready in {(time.time()-t)/60:.1f} min")


def fetch_records(con, table, ids):
    if isinstance(ids, pd.Series):
        ids = ids.tolist()
    elif isinstance(ids, np.ndarray):
        ids = ids.tolist()
    elif ids is None:
        ids = []
    elif not isinstance(ids, (list, tuple)):
        ids = [ids]
    clean_ids = []
    for x in ids:
        if x is None:
            continue
        x = str(x).strip()
        if x:
            clean_ids.append(x)
    if len(clean_ids) == 0:
        return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    ids_df = pd.DataFrame({"entity_id": pd.unique(np.asarray(clean_ids, dtype=object))})
    con.register("_ids", ids_df)
    try:
        return con.execute(f"""
            SELECT s.entity_id, s.business_name, s.business_address, s.country
            FROM {table} s
            INNER JOIN _ids i ON s.entity_id=i.entity_id
        """).fetchdf()
    finally:
        con.unregister("_ids")


def prepare_records(df):
    if df.empty:
        return df
    df = df.copy()
    df["name_norm"] = normalize_series(df["business_name"])
    df["address_norm"] = normalize_series(df["business_address"])
    df["country_norm"] = df["country"].fillna("").astype(str).str.lower().str.strip()
    df["numbers"] = df["address_norm"].map(lambda x: tuple(set(NUMBER_RE.findall(x))))
    return df.set_index("entity_id", drop=False)


def make_pairs(s1_df, cand_df, source_records):
    if cand_df.empty or source_records.empty:
        return pd.DataFrame(columns=["sid", "cid", "block_rank"])
    valid_ids = set(source_records.index.astype(str))
    rows = []
    sid_values = cand_df["source1_entity_id"].astype(str).tolist()
    candidate_lists = cand_df["candidate_entity_ids"].fillna("").astype(str).map(
        lambda x: [z.strip() for z in x.split("|") if z.strip()]
    )
    for sid, cands in zip(sid_values, candidate_lists):
        if sid not in s1_df.index:
            continue
        for block_rank, raw_cid in enumerate(cands, 1):
            cid = str(raw_cid).strip()
            if cid in valid_ids:
                rows.append((sid, cid, block_rank))
    if not rows:
        return pd.DataFrame(columns=["sid", "cid", "block_rank"])
    pairs = pd.DataFrame(rows, columns=["sid", "cid", "block_rank"])
    return pairs.drop_duplicates(["sid", "cid"], keep="first").reset_index(drop=True)


def add_base_features(pairs, s1, src):
    if pairs.empty:
        return pairs
    a = s1.loc[pairs.sid]
    b = src.loc[pairs.cid]
    qn = a["name_norm"].to_numpy(dtype=object)
    cn = b["name_norm"].to_numpy(dtype=object)
    qa = a["address_norm"].to_numpy(dtype=object)
    ca = b["address_norm"].to_numpy(dtype=object)
    pairs["name_ratio"] = process.cpdist(qn, cn, scorer=fuzz.ratio, dtype="float32", workers=-1)
    pairs["name_partial_ratio"] = process.cpdist(qn, cn, scorer=fuzz.partial_ratio, dtype="float32", workers=-1)
    pairs["name_token_set_ratio"] = process.cpdist(qn, cn, scorer=fuzz.token_set_ratio, dtype="float32", workers=-1)
    pairs["address_ratio"] = process.cpdist(qa, ca, scorer=fuzz.ratio, dtype="float32", workers=-1)
    pairs["address_token_set_ratio"] = process.cpdist(qa, ca, scorer=fuzz.token_set_ratio, dtype="float32", workers=-1)
    pairs["country_match"] = (
        a["country_norm"].to_numpy(dtype=object) == b["country_norm"].to_numpy(dtype=object)
    ).astype(np.int8)
    pairs["country_match"] &= (a["country_norm"].to_numpy(dtype=object) != "").astype(np.int8)
    an = a["numbers"].to_numpy(dtype=object)
    bn = b["numbers"].to_numpy(dtype=object)
    number_count = np.fromiter((len(set(x).intersection(y)) for x, y in zip(an, bn)), dtype=np.int16, count=len(pairs))
    pairs["address_number_count"] = number_count
    pairs["address_number_match"] = (number_count > 0).astype(np.int8)
    pairs["candidate_score"] = (
        0.45 * pairs["name_token_set_ratio"]
        + 0.20 * pairs["name_ratio"]
        + 0.15 * pairs["name_partial_ratio"]
        + 0.10 * pairs["address_token_set_ratio"]
        + 0.05 * pairs["address_ratio"]
        + 5.0 * pairs["country_match"]
        + 3.0 * pairs["address_number_match"]
    ).astype("float32")
    pairs["candidate_rank"] = (
        pairs.groupby("sid")["candidate_score"]
        .rank(method="first", ascending=False)
        .astype("int16")
    )
    return pairs


def keep_top_k(pairs, k=TOP_K_FOR_MODEL):
    if pairs.empty:
        return pairs
    return (pairs.sort_values(["sid", "candidate_score", "cid"], ascending=[True, False, True])
            .groupby("sid", sort=False, group_keys=False).head(k).copy())


def add_remaining_features(pairs, s1, src):
    if pairs.empty:
        return pairs
    a = s1.loc[pairs.sid]
    b = src.loc[pairs.cid]
    qn, cn = a["name_norm"].to_numpy(dtype=object), b["name_norm"].to_numpy(dtype=object)
    qa, ca = a["address_norm"].to_numpy(dtype=object), b["address_norm"].to_numpy(dtype=object)
    pairs["name_token_sort_ratio"] = process.cpdist(qn, cn, scorer=fuzz.token_sort_ratio, dtype="float32", workers=-1)
    pairs["address_partial_ratio"] = process.cpdist(qa, ca, scorer=fuzz.partial_ratio, dtype="float32", workers=-1)
    pairs["name_exact"] = (qn == cn).astype(np.int8) & (qn != "").astype(np.int8)
    pairs["address_exact"] = (qa == ca).astype(np.int8) & (qa != "").astype(np.int8)
    pairs["name_length_diff"] = np.abs(np.fromiter((len(x) for x in qn), dtype=np.int32, count=len(qn)) -
                                         np.fromiter((len(x) for x in cn), dtype=np.int32, count=len(cn)))
    pairs["address_length_diff"] = np.abs(np.fromiter((len(x) for x in qa), dtype=np.int32, count=len(qa)) -
                                            np.fromiter((len(x) for x in ca), dtype=np.int32, count=len(ca)))
    qt = a["name_norm"].map(lambda x: set(x.split())).to_numpy(dtype=object)
    ct = b["name_norm"].map(lambda x: set(x.split())).to_numpy(dtype=object)
    shared = np.fromiter((len(x.intersection(y)) for x, y in zip(qt, ct)), dtype=np.int16, count=len(pairs))
    qcount = np.fromiter((len(x) for x in qt), dtype=np.int16, count=len(qt))
    pairs["shared_token_count"] = shared
    pairs["shared_token_ratio"] = np.divide(shared, qcount, out=np.zeros(len(shared), dtype=np.float32), where=qcount > 0)
    return pairs


def main():
    print("=" * 78)
    print("MODEL REJECTION DIAGNOSTIC")
    print("=" * 78)

    # Load S1 sample
    print("\n[1] Loading S1 sample...")
    s1_all = pd.read_csv(S1_FILE, sep="\t", dtype=str, keep_default_na=False, na_filter=False, nrows=N_S1_SAMPLES)
    s1_all["entity_id"] = s1_all["entity_id"].astype(str)
    print(f"  S1 entities: {len(s1_all):,}")

    # Load ground truth
    print("\n[2] Loading ground truth...")
    gt = pd.read_csv(GT_FILE, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    s1_ids = set(s1_all["entity_id"])
    gt = gt[gt["source1_entity_id"].isin(s1_ids)].copy()
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
    print("\n[3] Loading V6 blocking candidates...")
    s2_cand = pd.read_csv(PROCESSED / "blocking_pilot_s1_s2_v4.csv", sep=",", dtype=str,
                         keep_default_na=False, na_filter=False)
    s3_cand = pd.read_csv(PROCESSED / "blocking_pilot_s1_s3_v4.csv", sep=",", dtype=str,
                         keep_default_na=False, na_filter=False)
    s2_cand = s2_cand[s2_cand["source1_entity_id"].isin(s1_ids)].copy()
    s3_cand = s3_cand[s3_cand["source1_entity_id"].isin(s1_ids)].copy()

    # Build candidate dict
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

    # Build features and get model probabilities
    print("\n[4] Building features and getting model probabilities...")
    model = load_model()
    con = duckdb.connect(str(PROCESSED / "training_validation.duckdb"))
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")
    build_or_open_cache(con)

    # Collect all pairs with their probabilities
    all_probs = []  # (sid, cid, prob, is_gt)
    gt_set = set()
    for s1_id, matches in gt_lookup.items():
        for m in matches:
            gt_set.add((s1_id, m))

    for chunk_start in range(0, len(s1_all), CHUNK_SIZE):
        chunk = s1_all.iloc[chunk_start:chunk_start + CHUNK_SIZE].copy()
        chunk_ids = set(chunk["entity_id"])

        s2_chunk = s2_cand[s2_cand["source1_entity_id"].isin(chunk_ids)]
        s3_chunk = s3_cand[s3_cand["source1_entity_id"].isin(chunk_ids)]

        s1 = chunk.copy()
        s1["name_norm"] = normalize_series(s1["business_name"])
        s1["address_norm"] = normalize_series(s1["business_address"])
        s1["country_norm"] = s1["country"].fillna("").astype(str).str.lower().str.strip()
        s1["numbers"] = s1["address_norm"].map(lambda x: tuple(set(NUMBER_RE.findall(x))))
        s1 = s1.set_index("entity_id", drop=False)

        s2_ids = s2_chunk["candidate_entity_ids"].fillna("").astype(str).map(
            lambda x: [z.strip() for z in x.split("|") if z.strip()]
        )
        s3_ids = s3_chunk["candidate_entity_ids"].fillna("").astype(str).map(
            lambda x: [z.strip() for z in x.split("|") if z.strip()]
        )

        s2_all_ids = [x for lst in s2_ids.tolist() for x in lst]
        s3_all_ids = [x for lst in s3_ids.tolist() for x in lst]

        s2 = prepare_records(fetch_records(con, "s2", s2_all_ids))
        s3 = prepare_records(fetch_records(con, "s3", s3_all_ids))

        p2 = make_pairs(s1, s2_chunk, s2)
        p3 = make_pairs(s1, s3_chunk, s3)

        p2 = keep_top_k(add_base_features(p2, s1, s2))
        p3 = keep_top_k(add_base_features(p3, s1, s3))

        # Add remaining features and predict
        for pairs, src in ((p2, s2), (p3, s3)):
            if pairs.empty:
                continue
            pairs = add_remaining_features(pairs, s1, src)
            X = pairs[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)
            probs = model.predict_proba(X)[:, 1]
            pairs["probability"] = probs

            for _, row in pairs.iterrows():
                sid = row["sid"]
                cid = row["cid"]
                prob = row["probability"]
                is_gt = (sid, cid) in gt_set
                all_probs.append((sid, cid, prob, is_gt))

        print(f"  Processed {min(chunk_start + CHUNK_SIZE, len(s1_all)):,}/{len(s1_all):,} S1")

    con.close()

    # Analyze probabilities
    print("\n[5] Analyzing probabilities...")
    probs_df = pd.DataFrame(all_probs, columns=["sid", "cid", "prob", "is_gt"])
    print(f"  Total pairs: {len(probs_df):,}")
    print(f"  GT pairs in candidates: {probs_df['is_gt'].sum():,}")

    # Probability distribution for GT pairs
    gt_probs = probs_df[probs_df["is_gt"]]["prob"]
    non_gt_probs = probs_df[~probs_df["is_gt"]]["prob"]

    print(f"\n  GT pairs probability distribution:")
    print(f"    Min:    {gt_probs.min():.4f}")
    print(f"    25th:   {gt_probs.quantile(0.25):.4f}")
    print(f"    Median: {gt_probs.median():.4f}")
    print(f"    75th:   {gt_probs.quantile(0.75):.4f}")
    print(f"    Max:    {gt_probs.max():.4f}")
    print(f"    Mean:   {gt_probs.mean():.4f}")

    print(f"\n  Non-GT pairs probability distribution:")
    print(f"    Min:    {non_gt_probs.min():.4f}")
    print(f"    25th:   {non_gt_probs.quantile(0.25):.4f}")
    print(f"    Median: {non_gt_probs.median():.4f}")
    print(f"    75th:   {non_gt_probs.quantile(0.75):.4f}")
    print(f"    Max:    {non_gt_probs.max():.4f}")
    print(f"    Mean:   {non_gt_probs.mean():.4f}")

    # Threshold analysis
    print(f"\n  Threshold analysis:")
    print(f"  {'Threshold':>10} {'TP':>8} {'FP':>8} {'FN':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        tp = ((probs_df["prob"] >= threshold) & (probs_df["is_gt"])).sum()
        fp = ((probs_df["prob"] >= threshold) & (~probs_df["is_gt"])).sum()
        fn = ((probs_df["prob"] < threshold) & (probs_df["is_gt"])).sum()
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        print(f"  {threshold:>10.1f} {tp:>8,} {fp:>8,} {fn:>8,} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")

    # Multi-match analysis
    print(f"\n[6] Multi-match analysis...")
    gt_counts = probs_df[probs_df["is_gt"]].groupby("sid").size()
    print(f"  GT matches per S1:")
    print(f"    1:  {(gt_counts == 1).sum():,}")
    print(f"    2:  {(gt_counts == 2).sum():,}")
    print(f"    3:  {(gt_counts == 3).sum():,}")
    print(f"    4+: {(gt_counts >= 4).sum():,}")

    # Check how many GT pairs are in top-k
    print(f"\n[7] Top-k analysis...")
    for k in [1, 5, 10, 20, 50]:
        top_k = probs_df.sort_values("prob", ascending=False).groupby("sid").head(k)
        gt_in_top_k = top_k["is_gt"].sum()
        total_gt = probs_df["is_gt"].sum()
        print(f"  Top-{k:>2}: {gt_in_top_k:>8,} / {total_gt:>8,} GT pairs recovered ({gt_in_top_k/total_gt:.2%})")

    # Feature importance
    print(f"\n[8] Feature importance...")
    booster = model.get_booster()
    importance = booster.get_score(importance_type="gain")
    for feat, score in sorted(importance.items(), key=lambda x: -x[1])[:10]:
        print(f"  {feat}: {score:.2f}")

    print("\n" + "=" * 78)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
