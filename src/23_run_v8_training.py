"""
23_run_v8_training.py

Run V8 blocking on training data to improve recall.
Uses the same V8 blocking rules but on training data.
"""

from pathlib import Path
import csv
import gc
import re
import sys
import time
from collections import defaultdict

import duckdb
import pandas as pd

ROOT = Path(r"C:\Users\adity\hackathon")
RAW_TRAIN = ROOT / "data" / "raw" / "train" / "train"
PROCESSED = ROOT / "data" / "processed"

DUCKDB_DIR = PROCESSED / "v8_duckdb_training"
DUCKDB_DIR.mkdir(parents=True, exist_ok=True)

S1_CHUNK_SIZE = 10_000
MAX_CANDIDATES_PER_S1 = 100
DUCKDB_MEMORY = "4GB"
DUCKDB_THREADS = 4

BLOCKING_RULES = {
    "exact_name": {"max_block_size": 5000},
    "token12": {"max_block_size": 2000},
    "token_number": {"max_block_size": 2000},
    "prefix5": {"max_block_size": 500},
    "suffix4": {"max_block_size": 1000},
    "last_token": {"max_block_size": 2000},
    "address_number": {"max_block_size": 500},
    "token_address": {"max_block_size": 1000},
    "token2_address": {"max_block_size": 1000},
    "address_token": {"max_block_size": 2000},
    "name_token_overlap": {"max_block_size": 5000},
    "address_token_overlap": {"max_block_size": 5000},
    "char_3gram_overlap": {"max_block_size": 5000},
}

RULE_WEIGHTS = {
    "exact_name": 1000,
    "token12": 350,
    "token_number": 320,
    "token_address": 280,
    "token2_address": 260,
    "prefix5": 220,
    "suffix4": 180,
    "last_token": 150,
    "address_number": 140,
    "address_token": 110,
    "name_token_overlap": 200,
    "address_token_overlap": 160,
    "char_3gram_overlap": 100,
}

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
NUMBER_RE = re.compile(r"\d{1,8}")


def normalize_text(x):
    if x is None or pd.isna(x):
        return ""
    x = str(x).lower().strip()
    if not x:
        return ""
    x = URL_RE.sub(" ", x)
    x = NON_WORD_RE.sub(" ", x)
    x = SPACE_RE.sub(" ", x).strip()
    return x


def tokenize(x):
    if not x:
        return []
    return [t for t in x.split() if len(t) >= 2]


def get_prefix(x, n=5):
    if not x:
        return ""
    compact = re.sub(r"\s+", "", x)
    return compact[:n] if len(compact) >= n else ""


def get_suffix(x, n=4):
    if not x:
        return ""
    compact = re.sub(r"\s+", "", x)
    return compact[-n:] if len(compact) >= n else ""


def get_char_ngrams(x, n=3):
    if not x:
        return set()
    compact = re.sub(r"\s+", "", x)
    if len(compact) < n:
        return set()
    return {compact[i:i+n] for i in range(len(compact) - n + 1)}


def build_duckdb_indexes(con, source_file, source_name):
    table = source_name
    src_path = str(source_file).replace("'", "''")

    existing = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name=?",
        [f"{table}_source"]
    ).fetchone()[0]

    if existing:
        count = con.execute(f"SELECT COUNT(*) FROM {table}_source").fetchone()[0]
        print(f"  [{source_name}] Index already exists: {count:,} records")
        return

    print(f"  [{source_name}] Building DuckDB indexes...")
    t0 = time.time()

    con.execute(f"""
        CREATE TABLE {table}_source AS
        SELECT
            entity_id,
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        lower(trim(business_name)),
                        'https?://\\S+|www\\.\\S+', ' ', 'g'
                    ),
                    '[^\\w\\s]', ' ', 'g'
                ),
                '\\s+', ' ', 'g'
            ) AS name_norm,
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        lower(trim(business_address)),
                        'https?://\\S+|www\\.\\S+', ' ', 'g'
                    ),
                    '[^\\w\\s]', ' ', 'g'
                ),
                '\\s+', ' ', 'g'
            ) AS address_norm,
            lower(trim(country)) AS country_norm
        FROM read_csv('{src_path}', delim='\t', header=true,
                      quote='"', escape='"', nullstr='',
                      columns={{
                        'entity_id': 'VARCHAR',
                        'business_name': 'VARCHAR',
                        'business_address': 'VARCHAR',
                        'country': 'VARCHAR'
                      }})
    """)

    n = con.execute(f"SELECT COUNT(*) FROM {table}_source").fetchone()[0]
    print(f"    {table}_source: {n:,} records")

    con.execute(f"""
        CREATE TABLE {table}_name_tokens AS
        SELECT DISTINCT s.entity_id, t.token, s.country_norm
        FROM {table}_source s
        CROSS JOIN UNNEST(string_split(s.name_norm, ' ')) AS t(token)
        WHERE length(t.token) >= 2
    """)
    n = con.execute(f"SELECT COUNT(*) FROM {table}_name_tokens").fetchone()[0]
    print(f"    {table}_name_tokens: {n:,} rows")

    con.execute(f"""
        CREATE TABLE {table}_addr_tokens AS
        SELECT DISTINCT s.entity_id, t.token, s.country_norm
        FROM {table}_source s
        CROSS JOIN UNNEST(string_split(s.address_norm, ' ')) AS t(token)
        WHERE length(t.token) >= 3
    """)
    n = con.execute(f"SELECT COUNT(*) FROM {table}_addr_tokens").fetchone()[0]
    print(f"    {table}_addr_tokens: {n:,} rows")

    con.execute(f"""
        CREATE TABLE {table}_char3grams AS
        SELECT DISTINCT s.entity_id, g.gram, s.country_norm
        FROM {table}_source s
        CROSS JOIN UNNEST(
            list_transform(
                range(greatest(length(regexp_replace(s.name_norm, '\\s+', '', 'g')) - 2, 0)),
                i -> substring(regexp_replace(s.name_norm, '\\s+', '', 'g'), i + 1, 3)
            )
        ) AS g(gram)
        WHERE length(regexp_replace(s.name_norm, '\\s+', '', 'g')) >= 3
    """)
    n = con.execute(f"SELECT COUNT(*) FROM {table}_char3grams").fetchone()[0]
    print(f"    {table}_char3grams: {n:,} rows")

    con.execute(f"CREATE INDEX idx_{table}_source_id ON {table}_source(entity_id)")
    con.execute(f"CREATE INDEX idx_{table}_nt_tok ON {table}_name_tokens(token, country_norm)")
    con.execute(f"CREATE INDEX idx_{table}_at_tok ON {table}_addr_tokens(token, country_norm)")
    con.execute(f"CREATE INDEX idx_{table}_c3g_gram ON {table}_char3grams(gram, country_norm)")
    con.execute(f"CREATE INDEX idx_{table}_src_name ON {table}_source(name_norm, country_norm)")

    con.execute("CHECKPOINT")
    elapsed = time.time() - t0
    print(f"  [{source_name}] Indexes built in {elapsed:.1f}s")


def prepare_s1_chunk(df):
    df = df.copy()
    df["name_norm"] = df["business_name"].map(normalize_text)
    df["address_norm"] = df["business_address"].map(normalize_text)
    df["country_norm"] = df["country"].map(normalize_text)
    df["tokens"] = df["name_norm"].map(tokenize)
    return df


def generate_candidates_for_chunk(con, s1_df, table):
    candidates = defaultdict(lambda: defaultdict(set))

    def run_rule(rule_name, df, query):
        tbl = f"_tmp_{rule_name}"
        con.register(tbl, df)
        try:
            hits = con.execute(query).fetchall()
            for s1_id, s2_id in hits:
                candidates[s1_id][s2_id].add(rule_name)
        except Exception as e:
            print(f"    [WARN] {rule_name} failed: {e}")
        finally:
            try:
                con.unregister(tbl)
            except:
                pass

    # Rule 1: exact_name
    s1_sub = s1_df[s1_df["name_norm"] != ""][["entity_id", "name_norm", "country_norm"]].copy()
    if not s1_sub.empty:
        run_rule("exact_name", s1_sub, f"""
            SELECT s.entity_id, src.entity_id
            FROM _tmp_exact_name s
            INNER JOIN {table}_source src
                ON src.name_norm = s.name_norm
                AND src.country_norm = s.country_norm
            WHERE s.country_norm != ''
        """)

    # Rule 2: token12
    s1_tok2 = s1_df[["entity_id", "country_norm", "tokens"]].copy()
    s1_tok2["token1"] = s1_tok2["tokens"].map(lambda x: x[0] if len(x) >= 1 else "")
    s1_tok2["token2"] = s1_tok2["tokens"].map(lambda x: x[1] if len(x) >= 2 else "")
    s1_tok2 = s1_tok2[(s1_tok2["token1"] != "") & (s1_tok2["token2"] != "")]
    s1_tok2 = s1_tok2[s1_tok2["country_norm"] != ""]
    if not s1_tok2.empty:
        run_rule("token12", s1_tok2, f"""
            SELECT DISTINCT s.entity_id, t1.entity_id
            FROM _tmp_token12 s
            INNER JOIN {table}_name_tokens t1
                ON t1.token = s.token1 AND t1.country_norm = s.country_norm
            INNER JOIN {table}_name_tokens t2
                ON t2.entity_id = t1.entity_id AND t2.token = s.token2 AND t2.country_norm = s.country_norm
            WHERE s.country_norm != ''
        """)

    # Rule 3: token_number
    s1_tn = s1_df[["entity_id", "country_norm", "tokens", "address_norm"]].copy()
    s1_tn["token1"] = s1_tn["tokens"].map(lambda x: x[0] if len(x) >= 1 else "")
    s1_tn["address_number"] = s1_tn["address_norm"].map(
        lambda x: next(iter(NUMBER_RE.findall(x)), "")
    )
    s1_tn = s1_tn[(s1_tn["token1"] != "") & (s1_tn["address_number"] != "")]
    s1_tn = s1_tn[s1_tn["country_norm"] != ""]
    if not s1_tn.empty:
        run_rule("token_number", s1_tn, f"""
            SELECT DISTINCT s.entity_id, t1.entity_id
            FROM _tmp_token_number s
            INNER JOIN {table}_name_tokens t1
                ON t1.token = s.token1 AND t1.country_norm = s.country_norm
            INNER JOIN {table}_source src
                ON src.entity_id = t1.entity_id
                AND src.address_norm LIKE '%' || s.address_number || '%'
            WHERE s.country_norm != ''
        """)

    # Rule 4: prefix5
    s1_pf = s1_df[["entity_id", "country_norm", "name_norm"]].copy()
    s1_pf["prefix5"] = s1_pf["name_norm"].map(lambda x: get_prefix(x, 5))
    s1_pf = s1_pf[s1_pf["prefix5"] != ""]
    s1_pf = s1_pf[s1_pf["country_norm"] != ""]
    if not s1_pf.empty:
        run_rule("prefix5", s1_pf, f"""
            SELECT DISTINCT s.entity_id, src.entity_id
            FROM _tmp_prefix5 s
            INNER JOIN {table}_source src
                ON src.name_norm LIKE s.prefix5 || '%'
                AND src.country_norm = s.country_norm
            WHERE s.country_norm != ''
        """)

    # Rule 5: suffix4
    s1_sf = s1_df[["entity_id", "country_norm", "name_norm"]].copy()
    s1_sf["suffix4"] = s1_sf["name_norm"].map(lambda x: get_suffix(x, 4))
    s1_sf = s1_sf[s1_sf["suffix4"] != ""]
    s1_sf = s1_sf[s1_sf["country_norm"] != ""]
    if not s1_sf.empty:
        run_rule("suffix4", s1_sf, f"""
            SELECT DISTINCT s.entity_id, src.entity_id
            FROM _tmp_suffix4 s
            INNER JOIN {table}_source src
                ON src.name_norm LIKE '%' || s.suffix4
                AND src.country_norm = s.country_norm
            WHERE s.country_norm != ''
        """)

    # Rule 6: last_token
    s1_lt = s1_df[["entity_id", "country_norm", "tokens"]].copy()
    s1_lt["last_token"] = s1_lt["tokens"].map(lambda x: x[-1] if x else "")
    s1_lt = s1_lt[s1_lt["last_token"] != ""]
    s1_lt = s1_lt[s1_lt["country_norm"] != ""]
    if not s1_lt.empty:
        run_rule("last_token", s1_lt, f"""
            SELECT DISTINCT s.entity_id, t1.entity_id
            FROM _tmp_last_token s
            INNER JOIN {table}_name_tokens t1
                ON t1.token = s.last_token AND t1.country_norm = s.country_norm
            WHERE s.country_norm != ''
        """)

    # Rule 7: address_number
    s1_an = s1_df[["entity_id", "country_norm", "address_norm"]].copy()
    s1_an["address_number"] = s1_an["address_norm"].map(
        lambda x: next(iter(NUMBER_RE.findall(x)), "")
    )
    s1_an = s1_an[s1_an["address_number"] != ""]
    s1_an = s1_an[s1_an["country_norm"] != ""]
    if not s1_an.empty:
        run_rule("address_number", s1_an, f"""
            SELECT DISTINCT s.entity_id, src.entity_id
            FROM _tmp_address_number s
            INNER JOIN {table}_source src
                ON src.address_norm LIKE '%' || s.address_number || '%'
                AND src.country_norm = s.country_norm
            WHERE s.country_norm != ''
        """)

    # Rule 8: token_address
    s1_ta = s1_df[["entity_id", "country_norm", "tokens", "address_norm"]].copy()
    s1_ta["token1"] = s1_ta["tokens"].map(lambda x: x[0] if len(x) >= 1 else "")
    s1_ta["addr_tokens"] = s1_ta["address_norm"].map(
        lambda x: [t for t in x.split() if len(t) >= 3]
    )
    s1_ta = s1_ta.explode("addr_tokens").reset_index(drop=True)
    s1_ta = s1_ta[(s1_ta["token1"] != "") & (s1_ta["addr_tokens"] != "")]
    s1_ta = s1_ta[s1_ta["country_norm"] != ""]
    if not s1_ta.empty:
        run_rule("token_address", s1_ta, f"""
            SELECT DISTINCT s.entity_id, t1.entity_id
            FROM _tmp_token_address s
            INNER JOIN {table}_name_tokens t1
                ON t1.token = s.token1 AND t1.country_norm = s.country_norm
            INNER JOIN {table}_addr_tokens at
                ON at.entity_id = t1.entity_id AND at.token = s.addr_tokens AND at.country_norm = s.country_norm
            WHERE s.country_norm != ''
        """)

    # Rule 9: token2_address
    s1_ta2 = s1_df[["entity_id", "country_norm", "tokens", "address_norm"]].copy()
    s1_ta2["token2"] = s1_ta2["tokens"].map(lambda x: x[1] if len(x) >= 2 else "")
    s1_ta2["addr_tokens"] = s1_ta2["address_norm"].map(
        lambda x: [t for t in x.split() if len(t) >= 3]
    )
    s1_ta2 = s1_ta2.explode("addr_tokens").reset_index(drop=True)
    s1_ta2 = s1_ta2[(s1_ta2["token2"] != "") & (s1_ta2["addr_tokens"] != "")]
    s1_ta2 = s1_ta2[s1_ta2["country_norm"] != ""]
    if not s1_ta2.empty:
        run_rule("token2_address", s1_ta2, f"""
            SELECT DISTINCT s.entity_id, t2.entity_id
            FROM _tmp_token2_address s
            INNER JOIN {table}_name_tokens t2
                ON t2.token = s.token2 AND t2.country_norm = s.country_norm
            INNER JOIN {table}_addr_tokens at
                ON at.entity_id = t2.entity_id AND at.token = s.addr_tokens AND at.country_norm = s.country_norm
            WHERE s.country_norm != ''
        """)

    # Rule 10: address_token
    s1_at = s1_df[["entity_id", "country_norm", "address_norm"]].copy()
    s1_at["addr_tokens"] = s1_at["address_norm"].map(
        lambda x: [t for t in x.split() if len(t) >= 4]
    )
    s1_at = s1_at.explode("addr_tokens").reset_index(drop=True)
    s1_at = s1_at[s1_at["addr_tokens"] != ""]
    s1_at = s1_at[s1_at["country_norm"] != ""]
    if not s1_at.empty:
        run_rule("address_token", s1_at, f"""
            SELECT DISTINCT s.entity_id, at.entity_id
            FROM _tmp_address_token s
            INNER JOIN {table}_addr_tokens at
                ON at.token = s.addr_tokens AND at.country_norm = s.country_norm
            WHERE s.country_norm != ''
        """)

    # Rule 11: name_token_overlap
    s1_nto = s1_df[["entity_id", "country_norm", "tokens"]].copy()
    s1_nto = s1_nto.explode("tokens").reset_index(drop=True)
    s1_nto = s1_nto[(s1_nto["tokens"] != "") & (s1_nto["country_norm"] != "")]
    if not s1_nto.empty:
        run_rule("name_token_overlap", s1_nto, f"""
            SELECT DISTINCT s.entity_id, t1.entity_id
            FROM _tmp_name_token_overlap s
            INNER JOIN {table}_name_tokens t1
                ON t1.token = s.tokens AND t1.country_norm = s.country_norm
            WHERE s.country_norm != ''
        """)

    # Rule 12: address_token_overlap
    s1_ato = s1_df[["entity_id", "country_norm", "address_norm"]].copy()
    s1_ato["addr_tokens"] = s1_ato["address_norm"].map(
        lambda x: [t for t in x.split() if len(t) >= 3]
    )
    s1_ato = s1_ato.explode("addr_tokens").reset_index(drop=True)
    s1_ato = s1_ato[(s1_ato["addr_tokens"] != "") & (s1_ato["country_norm"] != "")]
    if not s1_ato.empty:
        run_rule("address_token_overlap", s1_ato, f"""
            SELECT DISTINCT s.entity_id, at.entity_id
            FROM _tmp_address_token_overlap s
            INNER JOIN {table}_addr_tokens at
                ON at.token = s.addr_tokens AND at.country_norm = s.country_norm
            WHERE s.country_norm != ''
        """)

    # Rule 13: char_3gram_overlap
    s1_c3 = s1_df[["entity_id", "country_norm", "name_norm"]].copy()
    s1_c3["grams"] = s1_c3["name_norm"].map(lambda x: list(get_char_ngrams(x, 3)))
    s1_c3 = s1_c3.explode("grams").reset_index(drop=True)
    s1_c3 = s1_c3[(s1_c3["grams"] != "") & (s1_c3["country_norm"] != "")]
    if not s1_c3.empty:
        run_rule("char_3gram_overlap", s1_c3, f"""
            SELECT s1_id, s2_id
            FROM (
                SELECT
                    s.entity_id AS s1_id,
                    c.entity_id AS s2_id,
                    COUNT(DISTINCT s.grams) AS gram_overlap
                FROM _tmp_char_3gram_overlap s
                INNER JOIN {table}_char3grams c
                    ON c.gram = s.grams AND c.country_norm = s.country_norm
                WHERE s.country_norm != ''
                GROUP BY s.entity_id, c.entity_id
                HAVING COUNT(DISTINCT s.grams) >= 3
            )
        """)

    return candidates


def score_and_rank(candidates, max_candidates=MAX_CANDIDATES_PER_S1):
    scored = []
    for s1_id, cand_dict in candidates.items():
        for s2_id, rules in cand_dict.items():
            score = sum(RULE_WEIGHTS.get(r, 0) for r in rules)
            scored.append((s1_id, s2_id, score, rules))

    scored.sort(key=lambda x: (-x[2], x[0], x[1]))

    result = []
    seen_count = defaultdict(int)
    for s1_id, s2_id, score, rules in scored:
        if seen_count[s1_id] >= max_candidates:
            continue
        seen_count[s1_id] += 1
        result.append((s1_id, s2_id, score, rules))

    return result


def run_blocking(source_name, s1_limit=None, s1_chunk_size=S1_CHUNK_SIZE):
    print("=" * 72)
    print(f"V8 DISK-BACKED BLOCKER (TRAINING) -- S1 -> {source_name.upper()}")
    print("=" * 72)

    source_file = RAW_TRAIN / f"train_source{source_name[-1]}.tsv"
    s1_file = RAW_TRAIN / "train_source1.tsv"

    for f in [source_file, s1_file]:
        if not f.exists():
            raise FileNotFoundError(f)

    print(f"\nS1 file: {s1_file}")
    print(f"Source file: {source_file}")
    print(f"S1 chunk size: {s1_chunk_size:,}")
    print(f"S1 limit: {s1_limit if s1_limit else 'unlimited'}")

    duckdb_path = DUCKDB_DIR / f"v8_blocking_{source_name}.duckdb"
    con = duckdb.connect(str(duckdb_path))
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")

    print(f"\nDuckDB: {duckdb_path}")

    print(f"\n{'='*72}")
    print("PHASE 1: Building DuckDB indexes")
    print(f"{'='*72}")
    build_duckdb_indexes(con, source_file, source_name)

    print(f"\n{'='*72}")
    print("PHASE 2: Generating candidates")
    print(f"{'='*72}")

    output_dir = PROCESSED / "v8_output_training"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"blocking_v8_{source_name}.tsv"

    total_candidates = 0
    zero_candidates = 0
    max_candidates = 0
    rows_written = 0
    chunk_num = 0

    start_time = time.time()

    total_s1 = sum(1 for _ in open(s1_file, "r", encoding="utf-8")) - 1
    if s1_limit:
        total_s1 = min(total_s1, s1_limit)
    print(f"  Total S1 to process: {total_s1:,}")

    write_header = not output_file.exists()

    with open(output_file, "a", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t", lineterminator="\n")

        if write_header:
            writer.writerow([
                "source1_entity_id", "source_entity_id",
                "block_rank", "block_score", "block_rules",
            ])

        s1_reader = pd.read_csv(
            s1_file, sep="\t", dtype=str,
            keep_default_na=False, na_filter=False,
            chunksize=s1_chunk_size,
        )

        processed_this_run = 0

        for chunk in s1_reader:
            chunk_num += 1

            chunk = chunk.copy()
            if chunk.empty:
                continue

            s1_df = prepare_s1_chunk(chunk)

            candidates = generate_candidates_for_chunk(con, s1_df, source_name)

            ranked = score_and_rank(candidates)

            s1_groups = defaultdict(list)
            for s1_id, s2_id, score, rules in ranked:
                s1_groups[s1_id].append((s1_id, s2_id, score, rules))

            s1_ids_in_chunk = set(s1_df["entity_id"].astype(str))

            for s1_id in s1_ids_in_chunk:
                group = s1_groups.get(s1_id, [])
                count = len(group)

                total_candidates += count
                max_candidates = max(max_candidates, count)
                if count == 0:
                    zero_candidates += 1

                for rank, (sid, s2_id, score, rules) in enumerate(group, 1):
                    rule_string = ",".join(sorted(rules))
                    writer.writerow([sid, s2_id, rank, score, rule_string])
                    rows_written += 1

            processed_this_run += len(chunk)

            if chunk_num % 5 == 0:
                gc.collect()
                elapsed = time.time() - start_time
                rate = processed_this_run / max(elapsed, 1)
                print(f"  Chunk {chunk_num}: {processed_this_run:,}/{total_s1:,} "
                      f"| rate={rate:.1f}/s | cands={total_candidates:,} | "
                      f"zero={zero_candidates:,}")

            if s1_limit and processed_this_run >= s1_limit:
                break

    elapsed = time.time() - start_time
    print(f"\n  Total candidates: {total_candidates:,}")
    print(f"  Zero-candidate S1: {zero_candidates:,}")
    print(f"  Max candidates/S1: {max_candidates}")
    print(f"  Rows written: {rows_written:,}")
    print(f"  Elapsed: {elapsed:.1f}s")

    con.close()

    return output_file


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "s2"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if source not in {"s2", "s3"}:
        raise SystemExit("Usage: python 23_run_v8_training.py s2|s3 [limit]")

    output = run_blocking(source, s1_limit=limit)
    print(f"\nOutput: {output}")
