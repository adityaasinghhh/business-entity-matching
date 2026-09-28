"""
34_generate_candidate2.py

Generate improved submission using V6 candidates + additional targeted candidates.
Uses existing XGBoost model with threshold 0.50.
Outputs matching_results_candidate2.tsv (not matching_results.tsv).
"""

from pathlib import Path
import csv
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
RAW_TEST = ROOT / "data" / "raw" / "train" / "train" / "test" / "test"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUTPUT = ROOT / "outputs" / "final"

S1_FILE = RAW_TEST / "test_source1.tsv"
S2_FILE = RAW_TEST / "test_source2.tsv"
S3_FILE = RAW_TEST / "test_source3.tsv"

MODEL_FILE = MODELS / "xgboost_entity_match_v3.json"

MATCHING_OUTPUT = OUTPUT / "matching_results_candidate2.tsv"
CANDIDATE_OUTPUT = OUTPUT / "candidate_pairs_candidate2.tsv"

CHUNK_SIZE = 2500
TOP_K_FOR_MODEL = 20
MODEL_THRESHOLD = 0.50
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


def create_connection():
    con = duckdb.connect(str(PROCESSED / "test_source_lookup.duckdb"))
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")
    return con


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
        lambda x: [z.strip() for z in x.split(",") if z.strip()]
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


def predict_pairs(model, pairs, s1, src):
    if pairs.empty:
        return {}
    pairs = add_remaining_features(pairs, s1, src)
    X = pairs[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)
    probs = model.predict_proba(X)[:, 1]
    pairs["probability"] = probs

    results = {}
    for sid, g in pairs.groupby("sid", sort=False):
        g = g.sort_values("probability", ascending=False)
        accepted = g[g.probability >= MODEL_THRESHOLD]
        if len(accepted):
            results[sid] = accepted.cid.astype(str).tolist()
        else:
            results[sid] = []
    return results


def main():
    print("=" * 78)
    print("GENERATE IMPROVED SUBMISSION (candidate2)")
    print("=" * 78)

    for p in [S1_FILE, S2_FILE, S3_FILE, MODEL_FILE]:
        if not p.exists():
            raise FileNotFoundError(p)
        print(f"OK {p}")

    model = load_model()
    con = create_connection()

    # Load V6 blocking candidates
    print("\nLoading V6 blocking candidates...")
    s2_cand = pd.read_csv(PROCESSED / "blocking_test_s1_s2_v6.tsv", sep="\t", dtype=str,
                         keep_default_na=False, na_filter=False)
    s3_cand = pd.read_csv(PROCESSED / "blocking_test_s1_s3_v6.tsv", sep="\t", dtype=str,
                         keep_default_na=False, na_filter=False)
    print(f"  S2 candidates: {len(s2_cand):,} rows")
    print(f"  S3 candidates: {len(s3_cand):,} rows")

    # Process in chunks
    s1_iter = pd.read_csv(S1_FILE, sep="\t", dtype=str, keep_default_na=False,
                          na_filter=False, chunksize=CHUNK_SIZE)
    s2_iter = pd.read_csv(PROCESSED / "blocking_test_s1_s2_v6.tsv", sep="\t", dtype=str,
                          keep_default_na=False, na_filter=False, chunksize=CHUNK_SIZE)
    s3_iter = pd.read_csv(PROCESSED / "blocking_test_s1_s3_v6.tsv", sep="\t", dtype=str,
                          keep_default_na=False, na_filter=False, chunksize=CHUNK_SIZE)

    total = 0
    matched = 0
    empty = 0
    start = time.time()

    with open(MATCHING_OUTPUT, "w", encoding="utf-8", newline="") as mf, \
         open(CANDIDATE_OUTPUT, "w", encoding="utf-8", newline="") as cf:
        mw = csv.writer(mf, delimiter="\t", lineterminator="\n")
        cw = csv.writer(cf, delimiter="\t", lineterminator="\n")
        mw.writerow(["source1_entity_id", "matched_entity_ids"])
        cw.writerow(["source1_entity_id", "candidate_entity_ids"])

        for batch_no, (s1_raw, s2_c, s3_c) in enumerate(zip(s1_iter, s2_iter, s3_iter), 1):
            if len(s1_raw) != len(s2_c) or len(s1_raw) != len(s3_c):
                raise RuntimeError(f"Chunk length mismatch in batch {batch_no}")

            s1_raw = s1_raw.reset_index(drop=True)
            s2_c = s2_c.reset_index(drop=True)
            s3_c = s3_c.reset_index(drop=True)

            s1 = s1_raw.copy()
            s1["name_norm"] = normalize_series(s1["business_name"])
            s1["address_norm"] = normalize_series(s1["business_address"])
            s1["country_norm"] = s1["country"].fillna("").astype(str).str.lower().str.strip()
            s1["numbers"] = s1["address_norm"].map(lambda x: tuple(set(NUMBER_RE.findall(x))))
            s1 = s1.set_index("entity_id", drop=False)

            s2_ids = s2_c["candidate_entity_ids"].fillna("").astype(str).map(
                lambda x: [z.strip() for z in x.split(",") if z.strip()]
            )
            s3_ids = s3_c["candidate_entity_ids"].fillna("").astype(str).map(
                lambda x: [z.strip() for z in x.split(",") if z.strip()]
            )

            s2_all_ids = [x for lst in s2_ids.tolist() for x in lst]
            s3_all_ids = [x for lst in s3_ids.tolist() for x in lst]

            s2 = prepare_records(fetch_records(con, "s2", s2_all_ids))
            s3 = prepare_records(fetch_records(con, "s3", s3_all_ids))

            p2 = make_pairs(s1, s2_c, s2)
            p3 = make_pairs(s1, s3_c, s3)

            p2 = keep_top_k(add_base_features(p2, s1, s2))
            p3 = keep_top_k(add_base_features(p3, s1, s3))

            final_candidate_map = {}
            for df in (p2, p3):
                if not df.empty:
                    for sid_value, group in df.groupby("sid", sort=False):
                        sid_key = str(sid_value)
                        final_candidate_map.setdefault(sid_key, []).extend(
                            group["cid"].astype(str).tolist()
                        )

            r2 = predict_pairs(model, p2, s1, s2)
            r3 = predict_pairs(model, p3, s1, s3)

            for i, sid in enumerate(s1_raw["entity_id"].astype(str)):
                candidates = final_candidate_map.get(str(sid), [])
                candidates = list(dict.fromkeys(str(x).strip() for x in candidates if str(x).strip()))

                matches = list(r2.get(sid, [])) + list(r3.get(sid, []))
                matches = list(dict.fromkeys(str(x).strip() for x in matches if str(x).strip()))

                candidate_set = set(candidates)
                invalid_matches = [m for m in matches if m not in candidate_set]
                if invalid_matches:
                    raise RuntimeError(f"Output integrity failure for S1 {sid}")

                cw.writerow([sid, ",".join(candidates)])
                mw.writerow([sid, ",".join(matches)])
                total += 1
                if matches:
                    matched += 1
                else:
                    empty += 1

            elapsed = time.time() - start
            rate = total / elapsed if elapsed else 0
            print(f"\rProcessed {total:,} | rate={rate:.1f} S1/s | matched={matched:,} | empty={empty:,}", end="", flush=True)

    con.close()
    print("\n")

    # Validate
    expected = sum(1 for _ in open(S1_FILE, "r", encoding="utf-8")) - 1
    with open(MATCHING_OUTPUT, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        matching_rows = sum(1 for _ in reader)
    with open(CANDIDATE_OUTPUT, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        candidate_rows = sum(1 for _ in reader)

    print(f"\nExpected S1 rows: {expected:,}")
    print(f"matching rows: {matching_rows:,}")
    print(f"candidate rows: {candidate_rows:,}")

    if matching_rows != expected or candidate_rows != expected:
        raise RuntimeError("Row count mismatch")

    print("\nCOMPLETE")
    print(f"Matching: {MATCHING_OUTPUT}")
    print(f"Candidate: {CANDIDATE_OUTPUT}")


if __name__ == "__main__":
    main()
