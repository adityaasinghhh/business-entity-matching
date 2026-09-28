from pathlib import Path
import duckdb
import pandas as pd
import math

ROOT = Path(r"C:\Users\adity\hackathon")

GT = ROOT / r"data\raw\train\train\train_ground_truth.tsv"
MATCH = ROOT / r"outputs\final\matching_results.tsv"
CAND = ROOT / r"outputs\final\candidate_pairs.tsv"

print("=" * 75)
print("FINAL SUBMISSION — LOCAL VALIDATION")
print("=" * 75)

for p in [GT, MATCH, CAND]:
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    print("OK", p)

con = duckdb.connect()

con.execute(f"""
CREATE OR REPLACE TABLE gt AS
SELECT * FROM read_csv(
    '{str(GT).replace(chr(92), "/")}',
    delim='\t',
    header=true,
    all_varchar=true,
    nullstr=''
)
""")

con.execute(f"""
CREATE OR REPLACE TABLE mr AS
SELECT * FROM read_csv(
    '{str(MATCH).replace(chr(92), "/")}',
    delim='\t',
    header=true,
    all_varchar=true,
    nullstr=''
)
""")

con.execute(f"""
CREATE OR REPLACE TABLE cp AS
SELECT * FROM read_csv(
    '{str(CAND).replace(chr(92), "/")}',
    delim='\t',
    header=true,
    all_varchar=true,
    nullstr=''
)
""")

gt_cols = [r[0] for r in con.execute("DESCRIBE gt").fetchall()]
mr_cols = [r[0] for r in con.execute("DESCRIBE mr").fetchall()]
cp_cols = [r[0] for r in con.execute("DESCRIBE cp").fetchall()]

print("\nGround-truth columns :", gt_cols)
print("Matching columns     :", mr_cols)
print("Candidate columns    :", cp_cols)

# ------------------------------------------------------------
# BASIC COUNTS
# ------------------------------------------------------------

gt_rows = con.execute("SELECT COUNT(*) FROM gt").fetchone()[0]
mr_rows = con.execute("SELECT COUNT(*) FROM mr").fetchone()[0]
cp_rows = con.execute("SELECT COUNT(*) FROM cp").fetchone()[0]

print("\n" + "=" * 75)
print("ROW COUNTS")
print("=" * 75)
print(f"Ground truth rows : {gt_rows:,}")
print(f"Matching rows     : {mr_rows:,}")
print(f"Candidate rows    : {cp_rows:,}")

expected_s1 = 1_732_544

print(f"Expected test S1  : {expected_s1:,}")

if mr_rows == expected_s1:
    print("MATCHING ROW COUNT: PASS")
else:
    print("MATCHING ROW COUNT: FAIL")

if cp_rows == expected_s1:
    print("CANDIDATE ROW COUNT: PASS")
else:
    print("CANDIDATE ROW COUNT: FAIL")

# ------------------------------------------------------------
# S1 UNIQUENESS
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("S1 UNIQUENESS")
print("=" * 75)

mr_dup = con.execute("""
SELECT COUNT(*)
FROM (
    SELECT source1_entity_id
    FROM mr
    GROUP BY source1_entity_id
    HAVING COUNT(*) > 1
)
""").fetchone()[0]

cp_dup = con.execute("""
SELECT COUNT(*)
FROM (
    SELECT source1_entity_id
    FROM cp
    GROUP BY source1_entity_id
    HAVING COUNT(*) > 1
)
""").fetchone()[0]

print("Duplicate S1 IDs in matching_results :", mr_dup)
print("Duplicate S1 IDs in candidate_pairs :", cp_dup)

# ------------------------------------------------------------
# MATCH / EMPTY DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("MATCH DISTRIBUTION")
print("=" * 75)

dist = con.execute("""
SELECT
    SUM(
        CASE
            WHEN matched_entity_ids IS NULL
              OR TRIM(matched_entity_ids) = ''
            THEN 1 ELSE 0
        END
    ) AS empty_s1,
    SUM(
        CASE
            WHEN matched_entity_ids IS NOT NULL
             AND TRIM(matched_entity_ids) <> ''
            THEN 1 ELSE 0
        END
    ) AS matched_s1
FROM mr
""").fetchone()

empty_s1 = dist[0]
matched_s1 = dist[1]

print(f"S1 with >=1 match : {matched_s1:,}")
print(f"S1 with 0 matches : {empty_s1:,}")
print(f"Match coverage     : {matched_s1 / mr_rows * 100:.4f}%")

# ------------------------------------------------------------
# CANDIDATE DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("CANDIDATE DISTRIBUTION")
print("=" * 75)

cand_stats = con.execute("""
WITH x AS (
    SELECT
        source1_entity_id,
        CASE
            WHEN candidate_entity_ids IS NULL
              OR TRIM(candidate_entity_ids) = ''
            THEN 0
            ELSE array_length(
                string_split(candidate_entity_ids, ',')
            )
        END AS n
    FROM cp
)
SELECT
    MIN(n),
    AVG(n),
    MAX(n),
    SUM(CASE WHEN n = 0 THEN 1 ELSE 0 END),
    SUM(CASE WHEN n > 0 THEN 1 ELSE 0 END)
FROM x
""").fetchone()

print(f"Minimum candidates/S1 : {cand_stats[0]}")
print(f"Average candidates/S1 : {cand_stats[1]:.2f}")
print(f"Maximum candidates/S1 : {cand_stats[2]}")
print(f"S1 with zero          : {cand_stats[3]:,}")
print(f"S1 with candidates     : {cand_stats[4]:,}")

# ------------------------------------------------------------
# MATCHES MUST BE CANDIDATES
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("MATCH ⊆ CANDIDATES CHECK")
print("=" * 75)

# Explode matching IDs and candidates in DuckDB.
con.execute("""
CREATE OR REPLACE TABLE mr_exploded AS
SELECT
    source1_entity_id,
    TRIM(x) AS entity_id
FROM mr,
UNNEST(
    CASE
        WHEN matched_entity_ids IS NULL
          OR TRIM(matched_entity_ids) = ''
        THEN []
        ELSE string_split(matched_entity_ids, ',')
    END
) t(x)
WHERE TRIM(x) <> ''
""")

con.execute("""
CREATE OR REPLACE TABLE cp_exploded AS
SELECT
    source1_entity_id,
    TRIM(x) AS entity_id
FROM cp,
UNNEST(
    CASE
        WHEN candidate_entity_ids IS NULL
          OR TRIM(candidate_entity_ids) = ''
        THEN []
        ELSE string_split(candidate_entity_ids, ',')
    END
) t(x)
WHERE TRIM(x) <> ''
""")

invalid_match_count = con.execute("""
SELECT COUNT(*)
FROM mr_exploded m
LEFT JOIN cp_exploded c
  ON m.source1_entity_id = c.source1_entity_id
 AND m.entity_id = c.entity_id
WHERE c.entity_id IS NULL
""").fetchone()[0]

print("Predicted matches outside candidate set:", invalid_match_count)

if invalid_match_count == 0:
    print("MATCH ⊆ CANDIDATES: PASS")
else:
    print("MATCH ⊆ CANDIDATES: FAIL")

# ------------------------------------------------------------
# DUPLICATE MATCHES
# ------------------------------------------------------------

duplicate_matches = con.execute("""
SELECT COUNT(*)
FROM (
    SELECT source1_entity_id, entity_id
    FROM mr_exploded
    GROUP BY source1_entity_id, entity_id
    HAVING COUNT(*) > 1
)
""").fetchone()[0]

print("\nDuplicate predicted pairs:", duplicate_matches)

# ------------------------------------------------------------
# GROUND TRUTH SCHEMA
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("GROUND TRUTH ANALYSIS")
print("=" * 75)

# Automatically determine the source-ID column(s).
gt_id_cols = [
    c for c in gt_cols
    if c.lower() in {
        "source1_entity_id",
        "source2_entity_id",
        "source3_entity_id"
    }
]

print("Detected GT ID columns:", gt_id_cols)

if "source1_entity_id" not in gt_cols:
    print("\nWARNING: source1_entity_id was not found in ground truth.")
    print("Cannot calculate local Macro F0.5.")
else:
    # Build normalized ground-truth pair table.
    pair_queries = []

    for c in gt_id_cols:
        if c == "source1_entity_id":
            continue

        pair_queries.append(f"""
        SELECT
            CAST(source1_entity_id AS VARCHAR) AS sid,
            CAST({c} AS VARCHAR) AS cid
        FROM gt
        WHERE {c} IS NOT NULL
          AND TRIM(CAST({c} AS VARCHAR)) <> ''
        """)

    if pair_queries:
        union_sql = " UNION ALL ".join(pair_queries)

        con.execute(f"""
        CREATE OR REPLACE TABLE gt_pairs AS
        {union_sql}
        """)

        gt_pair_count = con.execute(
            "SELECT COUNT(*) FROM gt_pairs"
        ).fetchone()[0]

        print("Ground-truth positive pairs:", f"{gt_pair_count:,}")

        # ----------------------------------------------------
        # LOCAL MACRO F0.5
        # ----------------------------------------------------

        con.execute("""
        CREATE OR REPLACE TABLE per_s1 AS
        WITH s1 AS (
            SELECT DISTINCT source1_entity_id AS sid
            FROM mr
        ),
        truth AS (
            SELECT DISTINCT sid, cid
            FROM gt_pairs
        ),
        pred AS (
            SELECT DISTINCT source1_entity_id AS sid, entity_id AS cid
            FROM mr_exploded
        ),
        stats AS (
            SELECT
                s.sid,

                COUNT(DISTINCT CASE
                    WHEN p.cid IS NOT NULL
                     AND t.cid IS NOT NULL
                    THEN p.cid
                END) AS tp,

                COUNT(DISTINCT CASE
                    WHEN p.cid IS NOT NULL
                     AND t.cid IS NULL
                    THEN p.cid
                END) AS fp,

                COUNT(DISTINCT CASE
                    WHEN t.cid IS NOT NULL
                     AND p.cid IS NULL
                    THEN t.cid
                END) AS fn

            FROM s1 s
            LEFT JOIN pred p
                ON s.sid = p.sid
            FULL OUTER JOIN truth t
                ON s.sid = t.sid
               AND (
                    p.cid = t.cid
                    OR p.cid IS NULL
               )
            GROUP BY s.sid
        )
        SELECT * FROM stats
        """)

        # Use a simpler exact per-S1 computation through Python batches
        # because FULL OUTER JOIN above is only a diagnostic helper.
        rows = con.execute("""
        SELECT
            s.source1_entity_id AS sid,
            COALESCE(p.tp, 0) AS tp,
            COALESCE(p.fp, 0) AS fp,
            COALESCE(t.fn, 0) AS fn
        FROM
            (SELECT DISTINCT source1_entity_id FROM mr) s

        LEFT JOIN (
            SELECT
                m.source1_entity_id AS sid,
                SUM(CASE WHEN g.cid IS NOT NULL THEN 1 ELSE 0 END) AS tp,
                SUM(CASE WHEN g.cid IS NULL THEN 1 ELSE 0 END) AS fp
            FROM mr_exploded m
            LEFT JOIN (
                SELECT DISTINCT sid, cid
                FROM gt_pairs
            ) g
              ON m.source1_entity_id = g.sid
             AND m.entity_id = g.cid
            GROUP BY m.source1_entity_id
        ) p ON s.source1_entity_id = p.sid

        LEFT JOIN (
            SELECT
                g.sid,
                SUM(CASE WHEN m.entity_id IS NULL THEN 1 ELSE 0 END) AS fn
            FROM (
                SELECT DISTINCT sid, cid
                FROM gt_pairs
            ) g
            LEFT JOIN mr_exploded m
              ON g.sid = m.source1_entity_id
             AND g.cid = m.entity_id
            GROUP BY g.sid
        ) t ON s.source1_entity_id = t.sid
        """).fetchall()

        f05_values = []
        total_tp = total_fp = total_fn = 0

        for sid, tp, fp, fn in rows:
            tp = int(tp or 0)
            fp = int(fp or 0)
            fn = int(fn or 0)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 1.0

            if precision == 0 and recall == 0:
                f05 = 0.0
            else:
                f05 = (
                    1.25 * precision * recall /
                    (0.25 * precision + recall)
                )

            f05_values.append(f05)

        macro_f05 = sum(f05_values) / len(f05_values)

        print("\n" + "=" * 75)
        print("LOCAL MACRO F0.5")
        print("=" * 75)

        print(f"S1 evaluated : {len(f05_values):,}")
        print(f"TP           : {total_tp:,}")
        print(f"FP           : {total_fp:,}")
        print(f"FN           : {total_fn:,}")
        print(f"Micro precision: {total_tp/(total_tp+total_fp):.6f}")
        print(f"Micro recall   : {total_tp/(total_tp+total_fn):.6f}")
        print(f"MACRO F0.5     : {macro_f05:.6f}")
        print(f"MACRO F0.5 %   : {macro_f05*100:.4f}%")

        if macro_f05 >= 0.98:
            print("\nTARGET: >= 98% — CURRENT LOCAL RESULT PASSES")
        else:
            print("\nTARGET: >= 98% — CURRENT LOCAL RESULT BELOW TARGET")
            print("Further optimization should be considered before submission.")

print("\n" + "=" * 75)
print("VALIDATION COMPLETE")
print("=" * 75)

con.close()
