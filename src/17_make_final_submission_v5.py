from pathlib import Path
import csv
import re
import time
import os

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from rapidfuzz import process, fuzz

# ============================================================
# 17_make_final_submission_v5.py
# FAST V6 INFERENCE - SAME FILE, SAME MODEL, SAME OUTPUTS
# ============================================================

ROOT = Path(r"C:\Users\adity\hackathon")
DATA = ROOT / "data"
RAW_TEST = DATA / "raw" / "train" / "train" / "test" / "test"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
OUTPUT = ROOT / "outputs" / "final"
OUTPUT.mkdir(parents=True, exist_ok=True)

S1_FILE = RAW_TEST / "test_source1.tsv"
S2_FILE = RAW_TEST / "test_source2.tsv"
S3_FILE = RAW_TEST / "test_source3.tsv"

# IMPORTANT: these are the validated CURRENT V6 blockers.
S2_CANDIDATES = PROCESSED / "blocking_test_s1_s2_v6.tsv"
S3_CANDIDATES = PROCESSED / "blocking_test_s1_s3_v6.tsv"

MODEL_FILE = MODELS / "xgboost_entity_match_v3.json"

MATCHING_OUTPUT = OUTPUT / "matching_results.tsv"
CANDIDATE_OUTPUT = OUTPUT / "candidate_pairs.tsv"
CHECKPOINT_FILE = OUTPUT / "checkpoint.txt"

# A persistent DuckDB cache avoids rescanning 5M-row TSVs for every chunk.
CACHE_DB = PROCESSED / "test_source_lookup.duckdb"

# Current test-set row counts. These guards prevent an incompatible lookup
# cache from being silently reused.
EXPECTED_SOURCE_COUNTS = {
    "s2": 4_887_273,
    "s3": 5_082_316,
}

# Tuned for ~8 GB RAM. Increase to 5000 only if RAM remains comfortable.
CHUNK_SIZE = 2500
TOP_K_FOR_MODEL = 20
MODEL_THRESHOLD = 0.50
MIN_MARGIN = 0.05
USE_EXACT_NAME_RULE = False
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
    # Force object dtype so pandas does not route regex operations through
    # ArrowStringArray. This is more stable for the large inference loop.
    s = s.fillna("").astype(str).astype(object)
    return (s.str.lower().str.strip()
             .str.replace(URL_RE, " ", regex=True)
             .str.replace(NON_WORD_RE, " ", regex=True)
             .str.replace(SPACE_RE, " ", regex=True)
             .str.strip())


def normalize_text(x):
    if x is None:
        return ""
    x = str(x).lower().strip()
    if not x:
        return ""
    x = URL_RE.sub(" ", x)
    x = NON_WORD_RE.sub(" ", x)
    x = SPACE_RE.sub(" ", x)
    return x.strip()


def load_model():
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_FILE))
    names = model.get_booster().feature_names
    if names and list(names) != FEATURE_COLUMNS:
        raise RuntimeError(
            "Model feature order mismatch.\n"
            f"Model: {names}\nScript: {FEATURE_COLUMNS}"
        )
    return model


def create_connection():
    con = duckdb.connect(str(CACHE_DB))
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")
    return con


def qpath(path):
    return str(path).replace("'", "''")


def build_or_open_cache(con):
    """Build/reuse DuckDB source tables and reject incompatible row counts."""
    for table, path in (("s2", S2_FILE), ("s3", S3_FILE)):
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name=?",
            [table],
        ).fetchone()[0]

        if exists:
            cached_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            expected_count = EXPECTED_SOURCE_COUNTS[table]
            if cached_count == expected_count:
                print(f"Lookup cache: {table} ready ({cached_count:,} rows)")
                continue

            print(
                f"Lookup cache mismatch for {table}: "
                f"{cached_count:,} rows, expected {expected_count:,}. Rebuilding..."
            )
            con.execute(f"DROP TABLE {table}")
            con.execute("CHECKPOINT")

        print(f"Building lookup cache for {table} ...")
        t = time.time()
        con.execute(f"""
            CREATE TABLE {table} AS
            SELECT
                CAST(entity_id AS VARCHAR) AS entity_id,
                CAST(business_name AS VARCHAR) AS business_name,
                CAST(business_address AS VARCHAR) AS business_address,
                CAST(country AS VARCHAR) AS country
            FROM read_csv('{qpath(path)}', delim='\t', header=true,
                          quote='"', escape='"', nullstr='')
        """)
        actual_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        expected_count = EXPECTED_SOURCE_COUNTS[table]
        if actual_count != expected_count:
            raise RuntimeError(
                f"{table} source/cache row count mismatch after build: "
                f"{actual_count:,} != {expected_count:,}"
            )
        con.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_id ON {table}(entity_id)")
        con.execute("CHECKPOINT")
        print(f"  {table}: ready in {(time.time()-t)/60:.1f} min")



def fetch_records(con, table, ids):
    # Accept list / tuple / numpy array / pandas Series safely.
    if isinstance(ids, pd.Series):
        ids = ids.tolist()
    elif isinstance(ids, np.ndarray):
        ids = ids.tolist()
    elif ids is None:
        ids = []
    elif not isinstance(ids, (list, tuple)):
        ids = [ids]

    # Normalize and remove blanks before querying DuckDB.
    clean_ids = []
    for x in ids:
        if x is None:
            continue
        x = str(x).strip()
        if x:
            clean_ids.append(x)

    if len(clean_ids) == 0:
        return pd.DataFrame(
            columns=["entity_id", "business_name", "business_address", "country"]
        )

    ids_df = pd.DataFrame({
        "entity_id": pd.unique(np.asarray(clean_ids, dtype=object))
    })

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
    # Keep a compact number signature for fast intersection testing.
    df["numbers"] = df["address_norm"].map(lambda x: tuple(set(NUMBER_RE.findall(x))))
    return df.set_index("entity_id", drop=False)


def parse_candidate_series(s):
    """Parse candidate_entity_ids into one list per S1 row."""
    if isinstance(s, pd.DataFrame):
        if "candidate_entity_ids" not in s.columns:
            raise RuntimeError(
                "Candidate file is missing 'candidate_entity_ids' column."
            )
        s = s["candidate_entity_ids"]

    return s.fillna("").astype(str).map(
        lambda x: [z.strip() for z in x.split(",") if z.strip()]
    )


def make_pairs(s1_df, cand_df, source_records, source_prefix):
    """Flatten candidate lists and keep only IDs actually present in the source."""
    if cand_df.empty or source_records.empty:
        return pd.DataFrame(columns=["sid", "cid", "block_rank"])

    # Normalize the cache index once. This avoids fragile Index membership
    # behavior and is substantially faster than repeated `cid in index`.
    valid_ids = set(source_records.index.astype(str))

    rows = []
    sid_values = cand_df["source1_entity_id"].astype(str).tolist()
    candidate_lists = parse_candidate_series(cand_df["candidate_entity_ids"])

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
    # A candidate entity must appear only once in the final candidate set.
    return pairs.drop_duplicates(["sid", "cid"], keep="first").reset_index(drop=True)


def add_base_features(pairs, s1, src):
    """Vectorized/C++ RapidFuzz base features used for candidate_score."""
    if pairs.empty:
        return pairs

    a = s1.loc[pairs.sid]
    b = src.loc[pairs.cid]

    qn = a["name_norm"].to_numpy(dtype=object)
    cn = b["name_norm"].to_numpy(dtype=object)
    qa = a["address_norm"].to_numpy(dtype=object)
    ca = b["address_norm"].to_numpy(dtype=object)

    # cpdist computes aligned pairs in native RapidFuzz code.
    pairs["name_ratio"] = process.cpdist(qn, cn, scorer=fuzz.ratio, dtype="float32", workers=-1)
    pairs["name_partial_ratio"] = process.cpdist(qn, cn, scorer=fuzz.partial_ratio, dtype="float32", workers=-1)
    pairs["name_token_set_ratio"] = process.cpdist(qn, cn, scorer=fuzz.token_set_ratio, dtype="float32", workers=-1)
    pairs["address_ratio"] = process.cpdist(qa, ca, scorer=fuzz.ratio, dtype="float32", workers=-1)
    pairs["address_token_set_ratio"] = process.cpdist(qa, ca, scorer=fuzz.token_set_ratio, dtype="float32", workers=-1)

    pairs["country_match"] = (
        a["country_norm"].to_numpy(dtype=object) == b["country_norm"].to_numpy(dtype=object)
    ).astype(np.int8)
    pairs["country_match"] &= (a["country_norm"].to_numpy(dtype=object) != "").astype(np.int8)

    # Exact address-number intersection. Usually tiny tuples, so this is cheap.
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

    # Model was trained with candidate_rank after candidate scoring.
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

    # candidate_score/address_number fields were calculated before top-k.
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
        # Keep all matches above threshold (multi-match handling)
        accepted = g[g.probability >= MODEL_THRESHOLD]
        if len(accepted):
            results[sid] = accepted.cid.astype(str).tolist()
        else:
            results[sid] = []
    return results


def read_chunks(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       na_filter=False, chunksize=CHUNK_SIZE)


def validate_outputs():
    expected = sum(1 for _ in open(S1_FILE, "r", encoding="utf-8")) - 1

    with open(MATCHING_OUTPUT, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        matching_header = next(reader, None)
        matching_rows = sum(1 for _ in reader)

    with open(CANDIDATE_OUTPUT, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        candidate_header = next(reader, None)
        candidate_rows = sum(1 for _ in reader)

    print(f"\nExpected S1 rows : {expected:,}")
    print(f"matching rows    : {matching_rows:,}")
    print(f"candidate rows   : {candidate_rows:,}")

    if matching_header != ["source1_entity_id", "matched_entity_ids"]:
        raise RuntimeError(f"Invalid matching_results header: {matching_header}")

    if candidate_header != ["source1_entity_id", "candidate_entity_ids"]:
        raise RuntimeError(f"Invalid candidate_pairs header: {candidate_header}")

    if matching_rows != expected or candidate_rows != expected:
        raise RuntimeError(
            "Final output does not contain exactly one row per S1."
        )

    print("Row-count and header validation PASSED.")



def main():
    print("=" * 72)
    print("BUSINESS ENTITY MATCHING — FAST V6 INFERENCE (FINAL FIXED)")
    print("=" * 72)
    for p in [S1_FILE, S2_FILE, S3_FILE, S2_CANDIDATES, S3_CANDIDATES, MODEL_FILE]:
        if not p.exists():
            raise FileNotFoundError(p)
        print("OK", p)

    model = load_model()
    con = create_connection()
    build_or_open_cache(con)

    # ------------------------------------------------------------
    # PRE-FLIGHT SMOKE TEST
    # Never start a 1.7M-row run unless candidate lookup works.
    # ------------------------------------------------------------
    s1_test = pd.read_csv(
        S1_FILE, sep="\t", dtype=str, keep_default_na=False,
        na_filter=False, nrows=1
    )
    s2_test = pd.read_csv(
        S2_CANDIDATES, sep="\t", dtype=str, keep_default_na=False,
        na_filter=False, nrows=1
    )
    s3_test = pd.read_csv(
        S3_CANDIDATES, sep="\t", dtype=str, keep_default_na=False,
        na_filter=False, nrows=1
    )

    s1_test["entity_id"] = s1_test["entity_id"].astype(str)
    s1_test["name_norm"] = normalize_series(s1_test["business_name"])
    s1_test["address_norm"] = normalize_series(s1_test["business_address"])
    s1_test["country_norm"] = (
        s1_test["country"].fillna("").astype(str).str.lower().str.strip()
    )
    s1_test["numbers"] = s1_test["address_norm"].map(
        lambda x: tuple(set(NUMBER_RE.findall(x)))
    )
    s1_test = s1_test.set_index("entity_id", drop=False)

    s2_test_ids = parse_candidate_series(s2_test).iloc[0]
    s3_test_ids = parse_candidate_series(s3_test).iloc[0]

    test_s2 = prepare_records(fetch_records(con, "s2", s2_test_ids))
    test_s3 = prepare_records(fetch_records(con, "s3", s3_test_ids))

    print("\n--- PRE-FLIGHT SMOKE TEST ---")
    print(f"S2 candidate IDs : {len(s2_test_ids):,}")
    print(f"S2 records found : {len(test_s2):,}")
    print(f"S3 candidate IDs : {len(s3_test_ids):,}")
    print(f"S3 records found : {len(test_s3):,}")

    if len(s2_test_ids) > 0 and len(test_s2) == 0:
        con.close()
        raise RuntimeError(
            "SMOKE TEST FAILED: S2 candidate IDs were parsed but none "
            "were found in DuckDB."
        )
    if len(s3_test_ids) > 0 and len(test_s3) == 0:
        con.close()
        raise RuntimeError(
            "SMOKE TEST FAILED: S3 candidate IDs were parsed but none "
            "were found in DuckDB."
        )

    p2_test = make_pairs(s1_test, s2_test, test_s2, "S2")
    p3_test = make_pairs(s1_test, s3_test, test_s3, "S3")
    print(f"S2 pairs built     : {len(p2_test):,}")
    print(f"S3 pairs built     : {len(p3_test):,}")

    if (len(s2_test_ids) > 0 and len(p2_test) == 0) or (
        len(s3_test_ids) > 0 and len(p3_test) == 0
    ):
        con.close()
        raise RuntimeError(
            "SMOKE TEST FAILED: DuckDB records exist but make_pairs "
            "produced zero pairs."
        )

    # Run the complete feature + model path on this tiny sample.
    smoke_model_ok = False
    for smoke_pairs, smoke_src in ((p2_test, test_s2), (p3_test, test_s3)):
        if not smoke_pairs.empty:
            smoke_pairs = keep_top_k(add_base_features(smoke_pairs, s1_test, smoke_src))
            smoke_result = predict_pairs(model, smoke_pairs, s1_test, smoke_src)
            print(f"Model smoke predictions : {len(smoke_result):,}")
            smoke_model_ok = True
            break

    if not smoke_model_ok:
        con.close()
        raise RuntimeError("SMOKE TEST FAILED: no candidate pairs reached the model.")

    if len(s2_test_ids) != len(test_s2) or len(s3_test_ids) != len(test_s3):
        con.close()
        raise RuntimeError(
            "PRE-FLIGHT FAILED: not every first-row candidate ID was found "
            "in the DuckDB lookup tables."
        )

    print("PRE-FLIGHT SMOKE TEST PASSED.")
    print("--- starting full inference ---")

    # Candidate files and S1 are generated from the same S1 ordering.
    s1_iter = pd.read_csv(S1_FILE, sep="\t", dtype=str, keep_default_na=False,
                          na_filter=False, chunksize=CHUNK_SIZE)
    s2_iter = read_chunks(S2_CANDIDATES)
    s3_iter = read_chunks(S3_CANDIDATES)

    total = 0
    matched = 0
    empty = 0
    start = time.time()

    # Checkpoint: resume from last completed chunk
    resume_from = 0
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                resume_from = int(f.read().strip())
            print(f"RESUMING from S1 row {resume_from:,}")
        except Exception:
            resume_from = 0

    # Open in append mode if resuming, write mode if fresh
    write_mode = "a" if resume_from > 0 else "w"
    write_header = resume_from == 0

    with open(MATCHING_OUTPUT, write_mode, encoding="utf-8", newline="") as mf, \
         open(CANDIDATE_OUTPUT, write_mode, encoding="utf-8", newline="") as cf:
        mw = csv.writer(mf, delimiter="\t", lineterminator="\n")
        cw = csv.writer(cf, delimiter="\t", lineterminator="\n")

        if write_header:
            mw.writerow(["source1_entity_id", "matched_entity_ids"])
            cw.writerow(["source1_entity_id", "candidate_entity_ids"])
            mf.flush()
            cf.flush()

        for batch_no, (s1_raw, s2_cand, s3_cand) in enumerate(
            zip(s1_iter, s2_iter, s3_iter), 1
        ):
            chunk_start_row = (batch_no - 1) * CHUNK_SIZE

            # Skip already-completed chunks
            if chunk_start_row < resume_from:
                continue
            # zip() can silently truncate if one input has fewer rows.
            if len(s1_raw) != len(s2_cand) or len(s1_raw) != len(s3_cand):
                con.close()
                raise RuntimeError(
                    f"Input chunk length mismatch in batch {batch_no}: "
                    f"S1={len(s1_raw)}, S2-candidates={len(s2_cand)}, "
                    f"S3-candidates={len(s3_cand)}"
                )

            # Each pandas chunk retains the original global row index.
            # Reset all three chunk indexes before positional comparison.
            s1_raw = s1_raw.reset_index(drop=True)
            s2_cand = s2_cand.reset_index(drop=True)
            s3_cand = s3_cand.reset_index(drop=True)

            s1_ids = s1_raw["entity_id"].astype(str)
            s2_source_ids = s2_cand["source1_entity_id"].astype(str)
            s3_source_ids = s3_cand["source1_entity_id"].astype(str)

            if not s1_ids.equals(s2_source_ids) or not s1_ids.equals(s3_source_ids):
                bad_s2 = np.flatnonzero(
                    s1_ids.to_numpy() != s2_source_ids.to_numpy()
                )
                bad_s3 = np.flatnonzero(
                    s1_ids.to_numpy() != s3_source_ids.to_numpy()
                )
                raise RuntimeError(
                    f"S1/candidate ordering mismatch in batch {batch_no}. "
                    f"First S2 mismatch offset="
                    f"{int(bad_s2[0]) if len(bad_s2) else 'none'}, "
                    f"first S3 mismatch offset="
                    f"{int(bad_s3[0]) if len(bad_s3) else 'none'}"
                )
            s1 = s1_raw.copy()
            s1["name_norm"] = normalize_series(s1["business_name"])
            s1["address_norm"] = normalize_series(s1["business_address"])
            s1["country_norm"] = s1["country"].fillna("").astype(str).str.lower().str.strip()
            s1["numbers"] = s1["address_norm"].map(lambda x: tuple(set(NUMBER_RE.findall(x))))
            s1 = s1.set_index("entity_id", drop=False)

            s2_ids = parse_candidate_series(s2_cand)
            s3_ids = parse_candidate_series(s3_cand)

# Query each DuckDB source only with IDs belonging to that source.
            s2_all_ids = [x for lst in s2_ids.tolist() for x in lst]
            s3_all_ids = [x for lst in s3_ids.tolist() for x in lst]

            s2 = prepare_records(
                 fetch_records(con, "s2", s2_all_ids)
            )

            s3 = prepare_records(
                 fetch_records(con, "s3", s3_all_ids)
            )
            p2 = make_pairs(s1, s2_cand, s2, "S2")
            p3 = make_pairs(s1, s3_cand, s3, "S3")

            # Never silently process a large batch with missing lookups.
            requested_s2 = sum(len(x) for x in s2_ids)
            requested_s3 = sum(len(x) for x in s3_ids)

            if requested_s2 > 0 and p2.empty:
                print("\n")
                print("=" * 70)
                print("S2 LOOKUP DIAGNOSTIC")
                print("=" * 70)

                print(f"S1 rows in batch       : {len(s1):,}")
                print(f"S2 candidate IDs       : {requested_s2:,}")
                print(f"S2 records fetched     : {len(s2):,}")
                print(f"S2 pairs built         : {len(p2):,}")

                candidate_s2_ids = {
                    cid
                    for row_ids in s2_ids.tolist()
                    for cid in row_ids
                }

                fetched_s2_ids = set(s2.index.astype(str))
                overlap_s2 = candidate_s2_ids & fetched_s2_ids

                print(f"Unique S2 candidate IDs: {len(candidate_s2_ids):,}")
                print(f"Unique S2 fetched IDs  : {len(fetched_s2_ids):,}")
                print(f"S2 ID overlap          : {len(overlap_s2):,}")

                candidate_s1_ids = set(
                    s2_cand["source1_entity_id"].astype(str)
                )
                s1_ids_set = set(s1.index.astype(str))
                s1_overlap = candidate_s1_ids & s1_ids_set

                print(f"Candidate S1 IDs       : {len(candidate_s1_ids):,}")
                print(f"S1 IDs in batch        : {len(s1_ids_set):,}")
                print(f"S1 ID overlap          : {len(s1_overlap):,}")

                print("=" * 70)
                print("Candidate S2 example:")
                print(list(candidate_s2_ids)[:5])

                print("Fetched S2 example:")
                print(list(fetched_s2_ids)[:5])

                print("S2 overlap example:")
                print(list(overlap_s2)[:5])
                print("=" * 70)

                con.close()
                raise RuntimeError(
                    f"S2 LOOKUP FAILURE at S1 batch starting {total:,}"
                )

            if requested_s3 > 0 and p3.empty:
                con.close()
                raise RuntimeError(
                    f"S3 LOOKUP FAILURE at S1 batch starting {total:,}: "
                    f"{requested_s3:,} candidate IDs were present but zero "
                    f"S3 pairs were retrieved from DuckDB."
                )

            # These are the exact candidates that will be sent to XGBoost.
            # candidate_pairs.tsv must contain this final set, not the original
            # blocker candidate list.
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
                # candidate_pairs.tsv must contain the exact final candidate
                # set that was actually passed to XGBoost.
                candidates = final_candidate_map.get(str(sid), [])
                candidates = list(dict.fromkeys(str(x).strip() for x in candidates if str(x).strip()))

                matches = list(r2.get(sid, [])) + list(r3.get(sid, []))
                matches = list(dict.fromkeys(str(x).strip() for x in matches if str(x).strip()))

                # Every reported match must be present in candidate_pairs.tsv.
                candidate_set = set(candidates)
                invalid_matches = [m for m in matches if m not in candidate_set]
                if invalid_matches:
                    con.close()
                    raise RuntimeError(
                        f"Output integrity failure for S1 {sid}: "
                        f"match(es) not present in candidate_pairs: "
                        f"{invalid_matches[:5]}"
                    )

                cw.writerow([sid, ",".join(candidates)])
                mw.writerow([sid, ",".join(matches)])
                total += 1
                if matches:
                    matched += 1
                else:
                    empty += 1

            # Per-chunk logging
            chunk_end_row = chunk_start_row + len(s1_raw)
            elapsed = time.time() - start
            rate = total / elapsed if elapsed else 0
            print(f"Chunk {batch_no:>4} | S1 rows {chunk_start_row:>8,}-{chunk_end_row:>8,} | "
                  f"elapsed={elapsed:>7.1f}s | rate={rate:.1f} S1/s | "
                  f"total={total:,} | matched={matched:,} | empty={empty:,}", flush=True)

            # Write checkpoint
            with open(CHECKPOINT_FILE, "w") as f:
                f.write(str(chunk_end_row))

            # Flush output files
            mf.flush()
            cf.flush()

    con.close()
    print("\n")
    validate_outputs()
    print("\nCOMPLETE")
    print(MATCHING_OUTPUT)
    print(CANDIDATE_OUTPUT)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print("\n" + "=" * 78)
        print("ERROR: Script failed with exception")
        print("=" * 78)
        traceback.print_exc()
        print("=" * 78)
        sys.exit(1)
