import os
import gc
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

BASE = r"C:\Users\adity\hackathon"

FEATURE_FILE = os.path.join(
    BASE,
    "data",
    "processed",
    "features_pilot_s1_s2_v3.csv"
)

GT_FILE = os.path.join(
    BASE,
    "data",
    "raw",
    "train",
    "train",
    "train_ground_truth.tsv"
)

MODEL_FILE = os.path.join(
    BASE,
    "models",
    "xgboost_entity_match_v3.json"
)

CHUNK_SIZE = 250_000

RANDOM_STATE = 42

# Keep this conservative for your machine
N_ESTIMATORS = 300


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
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


# ============================================================
# START
# ============================================================

print("=" * 70)
print("XGBOOST ENTITY MATCHING - V3")
print("=" * 70)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "source1_entity_id",
        "matched_entity_ids"
    ]
)

print(
    f"Ground truth rows: {len(gt):,}"
)


# ============================================================
# CREATE POSITIVE PAIR SET
# ============================================================

print("\nCreating S1 -> S2 positive pairs...")

gt["matched_entity_ids"] = (
    gt["matched_entity_ids"]
    .fillna("")
)

gt["s2_matches"] = (
    gt["matched_entity_ids"]
    .str.split(",")
)

gt = gt[
    [
        "source1_entity_id",
        "s2_matches"
    ]
].explode(
    "s2_matches"
)

gt["source2_entity_id"] = (
    gt["s2_matches"]
    .fillna("")
    .str.strip()
)

gt = gt[
    gt["source2_entity_id"].str.startswith("S2-")
].copy()

gt = gt[
    [
        "source1_entity_id",
        "source2_entity_id"
    ]
].drop_duplicates()

print(
    f"S2 positive pairs: {len(gt):,}"
)


# ============================================================
# CREATE POSITIVE LOOKUP
# ============================================================

positive_pairs = set(
    zip(
        gt["source1_entity_id"],
        gt["source2_entity_id"]
    )
)

del gt
gc.collect()


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading feature data...")

parts = []

for chunk in pd.read_csv(
    FEATURE_FILE,
    dtype={
        "source1_entity_id": "string",
        "source2_entity_id": "string"
    },
    usecols=[
        "source1_entity_id",
        "source2_entity_id"
    ] + FEATURE_COLUMNS,
    chunksize=CHUNK_SIZE
):

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    pairs = list(
        zip(
            chunk["source1_entity_id"],
            chunk["source2_entity_id"]
        )
    )

    chunk["label"] = np.fromiter(
        (
            1 if pair in positive_pairs else 0
            for pair in pairs
        ),
        dtype=np.int8,
        count=len(chunk)
    )

    parts.append(chunk)

    print(
        f"Loaded: {sum(len(x) for x in parts):,}"
    )


data = pd.concat(
    parts,
    ignore_index=True
)

del parts
gc.collect()

print(
    f"\nTotal feature rows: {len(data):,}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

print(
    data["label"].value_counts()
)


positive_count = int(
    data["label"].sum()
)

negative_count = len(data) - positive_count

print(
    f"Positive: {positive_count:,}"
)

print(
    f"Negative: {negative_count:,}"
)


# ============================================================
# GROUP SPLIT
# ============================================================

print("\nCreating entity-level train/validation split...")

groups = data["source1_entity_id"].values

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_idx, valid_idx = next(
    splitter.split(
        data,
        data["label"],
        groups=groups
    )
)

train = data.iloc[train_idx].copy()
valid = data.iloc[valid_idx].copy()

del train_idx
del valid_idx
del groups
gc.collect()

print(
    f"Training rows: {len(train):,}"
)

print(
    f"Validation rows: {len(valid):,}"
)

print(
    f"Training S1 entities: "
    f"{train['source1_entity_id'].nunique():,}"
)

print(
    f"Validation S1 entities: "
    f"{valid['source1_entity_id'].nunique():,}"
)


# ============================================================
# X / Y
# ============================================================

X_train = train[FEATURE_COLUMNS].astype(
    np.float32
)

y_train = train["label"].astype(
    np.int8
)

X_valid = valid[FEATURE_COLUMNS].astype(
    np.float32
)

y_valid = valid["label"].astype(
    np.int8
)


# ============================================================
# CLASS WEIGHT
# ============================================================

positive_train = int(
    y_train.sum()
)

negative_train = (
    len(y_train) - positive_train
)

scale_pos_weight = (
    negative_train / positive_train
    if positive_train
    else 1.0
)

print(
    f"\nscale_pos_weight: "
    f"{scale_pos_weight:.2f}"
)


# ============================================================
# MODEL
# ============================================================

print("\nTraining XGBoost...")

model = XGBClassifier(

    n_estimators=N_ESTIMATORS,

    max_depth=6,

    learning_rate=0.08,

    min_child_weight=2,

    subsample=0.80,

    colsample_bytree=0.80,

    objective="binary:logistic",

    eval_metric="aucpr",

    tree_method="hist",

    max_bin=256,

    n_jobs=4,

    random_state=RANDOM_STATE,

    scale_pos_weight=scale_pos_weight,

    verbosity=1
)


model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_valid, y_valid)
    ],

    verbose=True
)


# ============================================================
# VALIDATION
# ============================================================

print("\nGenerating validation predictions...")

prob = model.predict_proba(
    X_valid
)[:, 1]

pred = (
    prob >= 0.50
).astype(
    np.int8
)


# ============================================================
# METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_valid,
    prob
)

pr_auc = average_precision_score(
    y_valid,
    prob
)

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(
    f"ROC-AUC: {roc_auc:.6f}"
)

print(
    f"PR-AUC : {pr_auc:.6f}"
)

print("\nClassification report:")

print(
    classification_report(
        y_valid,
        pred,
        digits=4
    )
)

print("\nConfusion matrix:")

print(
    confusion_matrix(
        y_valid,
        pred
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nFeature importance:")

importance = pd.Series(
    model.feature_importances_,
    index=FEATURE_COLUMNS
).sort_values(
    ascending=False
)

print(
    importance.to_string()
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(MODEL_FILE),
    exist_ok=True
)

model.save_model(
    MODEL_FILE
)

print(
    f"\nModel saved to:\n{MODEL_FILE}"
)

print("\nTraining complete.")