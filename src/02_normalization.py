from pathlib import Path
import re
import unicodedata

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "train"

SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_unicode(text):
    """
    Normalize Unicode without transliterating multilingual text.

    NFKC handles compatibility normalization while preserving
    scripts such as Hindi, Gujarati, Tamil, Kannada, etc.
    """

    if text is None:
        return ""

    text = str(text)

    if not text:
        return ""

    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text):
    """
    Collapse repeated whitespace into a single space.
    """

    if not text:
        return ""

    return re.sub(r"\s+", " ", text).strip()


def normalize_unicode_punctuation(text):
    """
    Replace punctuation and symbols with spaces while preserving:

    - Unicode letters
    - Unicode combining marks
    - Unicode numbers
    - Unicode whitespace

    This is specifically designed to avoid destroying Indic
    scripts such as:

        राम
        ગુજરાતી
        தமிழ்
        ಕನ್ನಡ
    """

    if not text:
        return ""

    result = []

    for char in text:

        category = unicodedata.category(char)

        # ----------------------------------------------------
        # Preserve whitespace
        # ----------------------------------------------------

        if char.isspace():
            result.append(" ")
            continue

        # ----------------------------------------------------
        # Preserve letters
        # ----------------------------------------------------

        if category.startswith("L"):
            result.append(char)
            continue

        # ----------------------------------------------------
        # Preserve combining marks
        #
        # Critical for Indic scripts.
        # ----------------------------------------------------

        if category.startswith("M"):
            result.append(char)
            continue

        # ----------------------------------------------------
        # Preserve numbers
        # ----------------------------------------------------

        if category.startswith("N"):
            result.append(char)
            continue

        # ----------------------------------------------------
        # Everything else is punctuation/symbol/control.
        # Replace with a space.
        # ----------------------------------------------------

        result.append(" ")

    return "".join(result)


# ============================================================
# BUSINESS NAME NORMALIZATION
# ============================================================

def normalize_name(text):
    """
    Normalize a business name.

    IMPORTANT:
    - Original Unicode scripts are preserved.
    - No transliteration.
    - No aggressive word removal.
    - Company suffixes are retained.
    - Numbers are retained.
    - Combining marks are retained.
    """

    if text is None:
        return ""

    # Unicode normalization
    text = normalize_unicode(text)

    if not text:
        return ""

    # Case normalization
    text = text.casefold()

    # --------------------------------------------------------
    # Remove URLs
    # --------------------------------------------------------

    text = re.sub(
        r"(?:https?://|www\.)[^\s]+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Unicode-safe punctuation normalization
    # --------------------------------------------------------

    text = normalize_unicode_punctuation(text)

    # --------------------------------------------------------
    # Whitespace normalization
    # --------------------------------------------------------

    text = normalize_whitespace(text)

    return text


# ============================================================
# BUSINESS ADDRESS NORMALIZATION
# ============================================================

def normalize_address(text):
    """
    Normalize a business address.

    IMPORTANT:
    We intentionally do NOT abbreviate:
        road -> rd
        street -> st
        etc.

    yet.

    That transformation will be evaluated separately against
    the ground truth later.
    """

    if text is None:
        return ""

    # Unicode normalization
    text = normalize_unicode(text)

    if not text:
        return ""

    # Case normalization
    text = text.casefold()

    # --------------------------------------------------------
    # Remove URLs
    # --------------------------------------------------------

    text = re.sub(
        r"(?:https?://|www\.)[^\s]+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Unicode-safe punctuation normalization
    # --------------------------------------------------------

    text = normalize_unicode_punctuation(text)

    # --------------------------------------------------------
    # Whitespace normalization
    # --------------------------------------------------------

    text = normalize_whitespace(text)

    return text


# ============================================================
# COUNTRY NORMALIZATION
# ============================================================

COUNTRY_MAP = {

    # --------------------------------------------------------
    # United States
    # --------------------------------------------------------

    "us": "us",
    "usa": "us",
    "united states": "us",
    "united states of america": "us",

    # --------------------------------------------------------
    # India
    # --------------------------------------------------------

    "in": "india",
    "ind": "india",
    "india": "india",
}


def normalize_country(text):
    """
    Convert common country representations into canonical values.
    """

    if text is None:
        return ""

    text = normalize_unicode(text)

    text = text.casefold()

    text = normalize_whitespace(text)

    return COUNTRY_MAP.get(text, text)


# ============================================================
# FIELD WRAPPERS
# ============================================================

def normalize_name_field(text):
    return normalize_name(text)


def normalize_address_field(text):
    return normalize_address(text)


# ============================================================
# DEBUG / NORMALIZATION TEST
# ============================================================

def test_normalization():

    name_examples = [
        "ABC Corporation Pvt. Ltd.",
        "ABC CORPORATION PVT LTD",
        "www.example.com Prime Money Inc.",
        "Delta Tetlecommunication Inc.",

        # Hindi
        "राम मार्केटिंग प्राइवेट लिमिटेड",

        # Gujarati
        "ગુજરાત બિઝનેસ સેન્ટર",

        # Tamil
        "தமிழ்நாடு மெடிக்கல் சென்டர்",

        # Kannada
        "ಕನ್ನಡ ಬಿಸಿನೆಸ್ ಸೆಂಟರ್",
    ]

    address_examples = [
        "1795 Westchester Drive, High Point, NC",
        "1795 Westchester Dr., High Point, NC",
        "105 ELM ST, MORGANTON, NC",
        "G-3/571, GULMOHAR COLONY, BHOPAL",

        # Gujarati
        "અમદાવાદ, ગુજરાત",

        # Kannada
        "ಬೆಂಗಳೂರು, ಕರ್ನಾಟಕ",
    ]

    country_examples = [
        "US",
        "USA",
        "United States",
        "United States of America",
        "India",
        "IN",
        "IND",
    ]

    print("\n" + "=" * 70)
    print("NORMALIZATION TEST")
    print("=" * 70)

    # ========================================================
    # BUSINESS NAMES
    # ========================================================

    print("\nBUSINESS NAMES")
    print("-" * 70)

    for value in name_examples:

        normalized = normalize_name_field(value)

        print(f"Original   : {value}")
        print(f"Normalized : {normalized}")

        # repr helps detect invisible Unicode/space problems
        print(f"DEBUG repr : {repr(normalized)}")

        print()

    # ========================================================
    # ADDRESSES
    # ========================================================

    print("\nADDRESSES")
    print("-" * 70)

    for value in address_examples:

        normalized = normalize_address_field(value)

        print(f"Original   : {value}")
        print(f"Normalized : {normalized}")
        print(f"DEBUG repr : {repr(normalized)}")

        print()

    # ========================================================
    # COUNTRIES
    # ========================================================

    print("\nCOUNTRIES")
    print("-" * 70)

    for value in country_examples:

        normalized = normalize_country(value)

        print(f"Original   : {value}")
        print(f"Normalized : {normalized}")
        print()


# ============================================================
# REAL DATA SAMPLE TEST
# ============================================================

def test_real_data():

    print("\n" + "=" * 70)
    print("REAL DATA SAMPLE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Read only 20 rows.
    # We are NOT processing millions of rows yet.
    # --------------------------------------------------------

    df = pd.read_csv(
        SOURCE1_PATH,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        nrows=20,
    )

    # --------------------------------------------------------
    # Normalize business names
    # --------------------------------------------------------

    df["name_normalized"] = (
        df["business_name"]
        .map(normalize_name_field)
    )

    # --------------------------------------------------------
    # Normalize addresses
    # --------------------------------------------------------

    df["address_normalized"] = (
        df["business_address"]
        .map(normalize_address_field)
    )

    # --------------------------------------------------------
    # Normalize countries
    # --------------------------------------------------------

    df["country_normalized"] = (
        df["country"]
        .map(normalize_country)
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    columns_to_show = [
        "business_name",
        "name_normalized",
        "business_address",
        "address_normalized",
        "country",
        "country_normalized",
    ]

    print(
        df[columns_to_show].to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUSINESS ENTITY MATCHING")
    print("PHASE 2 - NORMALIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Test manually defined examples
    # --------------------------------------------------------

    test_normalization()

    # --------------------------------------------------------
    # Test against actual Source-1 data
    # --------------------------------------------------------

    test_real_data()

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 2 SAMPLE TEST COMPLETE")
    print("=" * 70)

    print("\nNo full dataset was modified.")

    print(
        "\nBasic normalization rules are ready for review."
    )

    print(
        "\nNext step:"
        "\nValidate normalization against the ground truth "
        "before processing all 12.5M source records."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()