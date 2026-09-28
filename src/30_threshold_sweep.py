"""
30_threshold_sweep.py

Run threshold sweep on training data to find optimal threshold.
Uses 50K sample for speed.
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

N_S1_SAMPLES = 50_000
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
    print("THRESHOLD SWEEP ON TRAINING DATA")
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
    s1_with_gt = sum(1 for v in gt_lookup.values() if v)
    print(f"  GT pairs: {total_gt_pairs:,}")
    print(f"  S1 with GT: {s1_with_gt:,}")

    # Load blocking candidates
    print("\n[3] Loading V6 blocking candidates...")
    s2_cand = pd.read_csv(PROCESSED / "blocking_pilot_s1_s2_v4.csv", sep=",", dtype=str,
                         keep_default_na=False, na_filter=False)
    s3_cand = pd.read_csv(PROCESSED / "blocking_pilot_s1_s3_v4.csv", sep=",", dtype=str,
                         keep_default_na=False, na_filter=False)
    s2_cand = s2_cand[s2_cand["source1_entity_id"].isin(s1_ids)].copy()
    s3_cand = s3_cand[s3_cand["source1_entity_id"].isin(s1_ids)].copy()

    # Build features and get predictions
    print("\n[4] Building features and getting predictions...")
    model = load_model()
    con = duckdb.connect(str(PROCESSED / "training_validation.duckdb"))
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")
    build_or_open_cache(con)

    all_pairs_list = []

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

        # Add remaining features and get probabilities
        for pairs, src in ((p2, s2), (p3, s3)):
            if pairs.empty:
                continue
            pairs = add_remaining_features(pairs, s1, src)
            X = pairs[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)
            probs = model.predict_proba(X)[:, 1]
            pairs["probability"] = probs
            all_pairs_list.append(pairs)

        print(f"  Processed {min(chunk_start + CHUNK_SIZE, len(s1_all)):,}/{len(s1_all):,} S1")

    con.close()

    # Combine all pairs
    all_pairs = pd.concat(all_pairs_list, ignore_index=True)
    print(f"\n  Total pairs: {len(all_pairs):,}")

    # Threshold sweep
    print("\n[5] Threshold sweep...")
    print(f"  {'Threshold':>10} {'TP':>8} {'FP':>8} {'FN':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'F0.5':>8} {'EntRec':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        predictions = {}
        for sid, g in all_pairs.groupby("sid", sort=False):
            g = g.sort_values("probability", ascending=False)
            accepted = g[g.probability >= threshold]
            if len(accepted):
                predictions[sid] = accepted.cid.astype(str).tolist()
            else:
                predictions[sid] = []

        tp = fp = fn = 0
        entity_tp = entity_fn = 0
        f05_sum = 0.0

        for s1_id, gt_matches in gt_lookup.items():
            pred = set(predictions.get(s1_id, []))
            truth = gt_matches

            tp += len(pred & truth)
            fp += len(pred - truth)
            fn += len(truth - pred)

            if pred & truth:
                entity_tp += 1
            if truth - pred:
                entity_fn += 1

            # Per-entity F0.5
            p = len(pred & truth) / len(pred) if pred else 0.0
            r = len(pred & truth) / len(truth) if truth else 1.0
            if p == 0 and r == 0:
                f05 = 0.0
            else:
                f05 = 1.25 * p * r / (0.25 * p + r)
            f05_sum += f05

        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        f05 = 1.25 * prec * rec / (0.25 * prec + rec) if prec + rec else 0
        ent_rec = entity_tp / (entity_tp + entity_fn) if entity_tp + entity_fn else 0
        macro_f05 = f05_sum / len(gt_lookup) if gt_lookup else 0

        print(f"  {threshold:>10.2f} {tp:>8,} {fp:>8,} {fn:>8,} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f} {f05:>8.4f} {ent_rec:>8.4f}")

    print(f"\n  MACRO F0.5 (threshold 0.50): {macro_f05:.4f}")

    print("\n" + "=" * 78)
    print("THRESHOLD SWEEP COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
