import os
import duckdb

BASE = r"C:\Users\adity\hackathon"

S1_FILE = os.path.join(
    BASE, "data", "raw", "train", "train", "train_source1.tsv"
)

S2_FILE = os.path.join(
    BASE, "data", "raw", "train", "train", "train_source2.tsv"
)

BLOCK_FILE = os.path.join(
    BASE, "data", "processed", "blocking_pilot_s1_s2_v4.csv"
)

OUT_FILE = os.path.join(
    BASE, "data", "processed", "features_pilot_s1_s2_v4_fast.csv"
)

DB_FILE = os.path.join(
    BASE, "data", "processed", "entitymatch.duckdb"
)

TOP_K = 30

print("=" * 70)
print("ULTRA-FAST DUCKDB FEATURE GENERATION")
print("=" * 70)

con = duckdb.connect(DB_FILE)

con.execute("PRAGMA threads=6")
con.execute("PRAGMA memory_limit='5GB'")
con.execute("PRAGMA preserve_insertion_order=false")

# ------------------------------------------------------------
# Read Source 1
# ------------------------------------------------------------

print("\nLoading Source 1...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s1 AS
SELECT
    entity_id AS source1_entity_id,

    lower(
        regexp_replace(
            regexp_replace(
                coalesce(business_name, ''),
                'https?://[^ ]+|www\\.[^ ]+',
                '',
                'g'
            ),
            '[^[:alnum:]_ ]',
            ' ',
            'g'
        )
    ) AS name,

    lower(
        regexp_replace(
            coalesce(business_address, ''),
            '[^[:alnum:]_ ]',
            ' ',
            'g'
        )
    ) AS address,

    coalesce(country, '') AS country

FROM read_csv(
    '{S1_FILE.replace("\\", "/")}',
    delim='\\t',
    header=true,
    all_varchar=true
)
LIMIT 100000
""")

# ------------------------------------------------------------
# Read Source 2
# ------------------------------------------------------------

print("Loading Source 2...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE s2 AS
SELECT
    entity_id AS source2_entity_id,

    lower(
        regexp_replace(
            regexp_replace(
                coalesce(business_name, ''),
                'https?://[^ ]+|www\\.[^ ]+',
                '',
                'g'
            ),
            '[^[:alnum:]_ ]',
            ' ',
            'g'
        )
    ) AS name,

    lower(
        regexp_replace(
            coalesce(business_address, ''),
            '[^[:alnum:]_ ]',
            ' ',
            'g'
        )
    ) AS address,

    coalesce(country, '') AS country

FROM read_csv(
    '{S2_FILE.replace("\\", "/")}',
    delim='\\t',
    header=true,
    all_varchar=true
)
""")

# ------------------------------------------------------------
# Candidate pairs
#
# candidate_entity_ids are already ranked by blocking rules.
# We keep only TOP_K.
# ------------------------------------------------------------

print("Expanding V4 candidates...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE candidates AS

SELECT
    b.source1_entity_id,
    x.source2_entity_id,
    x.candidate_rank

FROM read_csv(
    '{BLOCK_FILE.replace("\\", "/")}',
    header=true,
    all_varchar=true
) b

CROSS JOIN LATERAL (
    SELECT
        value AS source2_entity_id,
        row_number() OVER () AS candidate_rank
    FROM unnest(
        string_split(b.candidate_entity_ids, '|')
    ) AS t(value)
) x

WHERE x.candidate_rank <= {TOP_K}
""")

count = con.execute(
    "SELECT COUNT(*) FROM candidates"
).fetchone()[0]

print(f"Candidate feature rows: {count:,}")

# ------------------------------------------------------------
# Join Source 1 / Source 2
# ------------------------------------------------------------

print("Joining entities...")

con.execute("""
CREATE OR REPLACE TEMP TABLE joined AS

SELECT
    c.source1_entity_id,
    c.source2_entity_id,
    c.candidate_rank,

    s1.name AS name1,
    s2.name AS name2,

    s1.address AS address1,
    s2.address AS address2,

    s1.country AS country1,
    s2.country AS country2

FROM candidates c

JOIN s1
    ON c.source1_entity_id = s1.source1_entity_id

JOIN s2
    ON c.source2_entity_id = s2.source2_entity_id
""")

# ------------------------------------------------------------
# FAST FEATURES
#
# No Python.
# No RapidFuzz.
# Uses DuckDB string/list operations.
# ------------------------------------------------------------

print("Computing fast features...")

query = f"""
COPY (

WITH base AS (

    SELECT
        *,

        string_split(
            regexp_replace(name1, '\\s+', ' ', 'g'),
            ' '
        ) AS name_tokens1,

        string_split(
            regexp_replace(name2, '\\s+', ' ', 'g'),
            ' '
        ) AS name_tokens2,

        string_split(
            regexp_replace(address1, '\\s+', ' ', 'g'),
            ' '
        ) AS address_tokens1,

        string_split(
            regexp_replace(address2, '\\s+', ' ', 'g'),
            ' '
        ) AS address_tokens2

    FROM joined
),

features AS (

    SELECT

        source1_entity_id,
        source2_entity_id,

        -- --------------------------------------------------
        -- Name similarity proxies
        -- --------------------------------------------------

        CASE
            WHEN name1 = name2 AND name1 <> '' THEN 100
            WHEN length(name1) = 0 OR length(name2) = 0 THEN 0
            ELSE
                100.0 *
                least(length(name1), length(name2))
                / greatest(length(name1), length(name2))
        END AS name_ratio,

        CASE
            WHEN name1 = name2 AND name1 <> '' THEN 100
            WHEN position(split_part(name1, ' ', 1) IN name2) > 0
                THEN 80
            ELSE 0
        END AS name_partial_ratio,

        CASE
            WHEN name1 = name2 THEN 100
            ELSE
                100.0 *
                list_count(
                    list_intersect(name_tokens1, name_tokens2)
                )
                /
                greatest(
                    1,
                    list_count(
                        list_distinct(
                            list_concat(name_tokens1, name_tokens2)
                        )
                    )
                )
        END AS name_token_sort_ratio,

        CASE
            WHEN name1 = name2 THEN 100
            ELSE
                100.0 *
                list_count(
                    list_intersect(name_tokens1, name_tokens2)
                )
                /
                greatest(
                    1,
                    least(
                        list_count(name_tokens1),
                        list_count(name_tokens2)
                    )
                )
        END AS name_token_set_ratio,

        CASE
            WHEN name1 = name2 AND name1 <> '' THEN 1
            ELSE 0
        END AS name_exact,

        abs(length(name1) - length(name2)) AS name_length_diff,

        -- --------------------------------------------------
        -- Address similarity
        -- --------------------------------------------------

        CASE
            WHEN address1 = address2 AND address1 <> '' THEN 100
            WHEN length(address1) = 0 OR length(address2) = 0 THEN 0
            ELSE
                100.0 *
                least(length(address1), length(address2))
                /
                greatest(length(address1), length(address2))
        END AS address_ratio,

        CASE
            WHEN address1 = address2 AND address1 <> '' THEN 100
            ELSE
                100.0 *
                list_count(
                    list_intersect(address_tokens1, address_tokens2)
                )
                /
                greatest(
                    1,
                    least(
                        list_count(address_tokens1),
                        list_count(address_tokens2)
                    )
                )
        END AS address_partial_ratio,

        CASE
            WHEN address1 = address2 THEN 100
            ELSE
                100.0 *
                list_count(
                    list_intersect(address_tokens1, address_tokens2)
                )
                /
                greatest(
                    1,
                    least(
                        list_count(address_tokens1),
                        list_count(address_tokens2)
                    )
                )
        END AS address_token_set_ratio,

        CASE
            WHEN address1 <> '' AND address1 = address2 THEN 1
            ELSE 0
        END AS address_exact,

        abs(length(address1) - length(address2))
            AS address_length_diff,

        -- --------------------------------------------------
        -- Country
        -- --------------------------------------------------

        CASE
            WHEN country1 <> ''
             AND country1 = country2
            THEN 1
            ELSE 0
        END AS country_match,

        -- --------------------------------------------------
        -- Shared name tokens
        -- --------------------------------------------------

        least(
            127,
            list_count(
                list_intersect(name_tokens1, name_tokens2)
            )
        ) AS shared_token_count,

        list_count(
            list_intersect(name_tokens1, name_tokens2)
        )
        /
        greatest(
            1,
            list_count(name_tokens1)
        ) AS shared_token_ratio,

        -- --------------------------------------------------
        -- Address number
        -- --------------------------------------------------

        CASE
            WHEN regexp_extract(address1, '[0-9]+')
               <> ''
             AND regexp_extract(address1, '[0-9]+')
                =
                 regexp_extract(address2, '[0-9]+')
            THEN 1
            ELSE 0
        END AS address_number_match,

        CASE
            WHEN regexp_extract(address1, '[0-9]+') <> ''
             AND regexp_extract(address1, '[0-9]+')
                =
                 regexp_extract(address2, '[0-9]+')
            THEN 1
            ELSE 0
        END AS address_number_count,

        -- --------------------------------------------------
        -- Candidate ranking proxy
        -- --------------------------------------------------

        candidate_rank,

        100.0 - candidate_rank AS candidate_score

    FROM base
)

SELECT
    source1_entity_id,
    source2_entity_id,
    CAST(name_ratio AS FLOAT),
    CAST(name_partial_ratio AS FLOAT),
    CAST(name_token_sort_ratio AS FLOAT),
    CAST(name_token_set_ratio AS FLOAT),
    CAST(name_exact AS TINYINT),
    CAST(name_length_diff AS SMALLINT),

    CAST(address_ratio AS FLOAT),
    CAST(address_partial_ratio AS FLOAT),
    CAST(address_token_set_ratio AS FLOAT),
    CAST(address_exact AS TINYINT),
    CAST(address_length_diff AS SMALLINT),

    CAST(country_match AS TINYINT),

    CAST(shared_token_count AS TINYINT),
    CAST(shared_token_ratio AS FLOAT),

    CAST(address_number_match AS TINYINT),
    CAST(address_number_count AS TINYINT),

    CAST(candidate_score AS FLOAT),
    CAST(candidate_rank AS INTEGER)

FROM features

) TO '{OUT_FILE.replace("\\", "/")}'
WITH (
    HEADER,
    DELIMITER ','
)
"""

con.execute(query)

print("\n" + "=" * 70)
print("FAST FEATURE GENERATION COMPLETE")
print("=" * 70)
print(f"Output: {OUT_FILE}")

size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024)

print(f"Size: {size_mb:.2f} MB")

con.close()