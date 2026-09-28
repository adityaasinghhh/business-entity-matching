from pathlib import Path
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "train"

SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"
GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"


# ============================================================
# 2. EXPECTED COLUMNS
# ============================================================

SOURCE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

GROUND_TRUTH_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]


# ============================================================
# 3. LOAD TSV
# ============================================================

def load_tsv(path: Path) -> pd.DataFrame:
    """
    Load a TSV file safely as strings.

    keep_default_na=False prevents empty strings from
    automatically becoming NaN.
    """

    print(f"\nLoading:")
    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        low_memory=False,
    )

    print(f"Loaded {len(df):,} rows")

    return df


# ============================================================
# 4. BASIC DATASET AUDIT
# ============================================================

def audit_source(df: pd.DataFrame, dataset_name: str):
    """
    Print basic information about a source dataset.
    """

    print("\n" + "=" * 70)
    print(f"DATASET AUDIT: {dataset_name}")
    print("=" * 70)

    # Shape
    print(f"\nRows       : {len(df):,}")
    print(f"Columns    : {len(df.columns)}")

    # Columns
    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    # Expected columns
    missing_columns = [
        column
        for column in SOURCE_COLUMNS
        if column not in df.columns
    ]

    unexpected_columns = [
        column
        for column in df.columns
        if column not in SOURCE_COLUMNS
    ]

    if missing_columns:
        print("\nWARNING - Missing expected columns:")
        for column in missing_columns:
            print(f"  - {column}")

    if unexpected_columns:
        print("\nAdditional columns:")
        for column in unexpected_columns:
            print(f"  - {column}")

    # Empty values
    print("\nEmpty values:")
    for column in df.columns:
        empty_count = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        percentage = (
            empty_count / len(df) * 100
            if len(df) > 0
            else 0
        )

        print(
            f"  {column:<20} "
            f"{empty_count:>10,} "
            f"({percentage:>6.2f}%)"
        )

    # Duplicate IDs
    if "entity_id" in df.columns:

        duplicate_id_count = df["entity_id"].duplicated().sum()

        unique_id_count = df["entity_id"].nunique()

        print("\nEntity ID statistics:")
        print(f"  Unique IDs          : {unique_id_count:,}")
        print(f"  Duplicate ID rows   : {duplicate_id_count:,}")

    # Duplicate names
    if "business_name" in df.columns:

        duplicate_name_count = (
            df["business_name"]
            .str.lower()
            .duplicated()
            .sum()
        )

        print("\nBusiness name statistics:")
        print(
            f"  Duplicate name rows : "
            f"{duplicate_name_count:,}"
        )

    # Country distribution
    if "country" in df.columns:

        print("\nTop countries:")

        country_counts = (
            df["country"]
            .value_counts()
            .head(15)
        )

        for country, count in country_counts.items():
            print(f"  {str(country):<20} {count:>10,}")

    # Sample
    print("\nFirst 5 rows:")
    print(
        df.head(5).to_string(index=False)
    )


# ============================================================
# 5. GROUND TRUTH AUDIT
# ============================================================

def audit_ground_truth(df: pd.DataFrame):
    """
    Analyze the ground-truth mapping.

    One Source-1 entity can have zero, one, or multiple
    matched entities.
    """

    print("\n" + "=" * 70)
    print("GROUND TRUTH AUDIT")
    print("=" * 70)

    print(f"\nRows: {len(df):,}")

    print("\nColumns:")

    for column in df.columns:
        print(f"  - {column}")

    # Check expected columns
    missing_columns = [
        column
        for column in GROUND_TRUTH_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print("\nWARNING - Missing expected columns:")

        for column in missing_columns:
            print(f"  - {column}")

    # Empty matches
    if "matched_entity_ids" in df.columns:

        empty_matches = (
            df["matched_entity_ids"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        )

        print(
            f"\nSource-1 entities with no matches: "
            f"{empty_matches.sum():,}"
        )

        # Number of matched entities per S1
        match_counts = (
            df["matched_entity_ids"]
            .fillna("")
            .astype(str)
            .str.strip()
            .apply(
                lambda x: 0
                if x == ""
                else len(x.split(","))
            )
        )

        print("\nNumber of matches per Source-1 entity:")

        print(
            match_counts
            .value_counts()
            .sort_index()
            .to_string()
        )

        print("\nMatch statistics:")

        print(
            f"  Zero matches : {(match_counts == 0).sum():,}"
        )

        print(
            f"  One match   : {(match_counts == 1).sum():,}"
        )

        print(
            f"  Multiple    : {(match_counts > 1).sum():,}"
        )

        print(
            f"  Maximum     : {match_counts.max():,}"
        )

    print("\nFirst 10 ground-truth rows:")

    print(
        df.head(10).to_string(index=False)
    )


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("BUSINESS ENTITY MATCHING")
    print("PHASE 1 - DATA LOADING & AUDIT")
    print("=" * 70)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print(f"\nTraining data directory:")
    print(TRAIN_DIR)

    # --------------------------------------------------------
    # Load source datasets
    # --------------------------------------------------------

    source1 = load_tsv(SOURCE1_PATH)
    source2 = load_tsv(SOURCE2_PATH)
    source3 = load_tsv(SOURCE3_PATH)

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    ground_truth = load_tsv(GROUND_TRUTH_PATH)

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    audit_source(
        source1,
        "SOURCE 1"
    )

    audit_source(
        source2,
        "SOURCE 2"
    )

    audit_source(
        source3,
        "SOURCE 3"
    )

    audit_ground_truth(
        ground_truth
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)

    print("\nAll four datasets were loaded successfully.")

    print("\nNext step:")
    print("We will analyze the audit results before building")
    print("the normalization pipeline.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()