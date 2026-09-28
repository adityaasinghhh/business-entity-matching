import os
import pandas as pd
import xgboost as xgb

BASE = r"C:\Users\adity\hackathon"

FEATURE_FILE = os.path.join(
    BASE, "data", "processed",
    "features_pilot_s1_s2_v3.csv"
)

MODEL_FILE = os.path.join(
    BASE, "models",
    "xgboost_entity_match_v3.json"
)

OUT_FILE = os.path.join(
    BASE, "outputs",
    "s1_s2_predictions.csv"
)

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

print("=" * 70)
print("S1 -> S2 PREDICTION")
print("=" * 70)

print("\nLoading features...")

df = pd.read_csv(
    FEATURE_FILE,
    dtype={
        "source1_entity_id": str,
        "source2_entity_id": str
    }
)

print(f"Feature rows: {len(df):,}")

FEATURES = [
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
    "candidate_rank"
]

print("\nLoading XGBoost model...")

model = xgb.XGBClassifier()
model.load_model(MODEL_FILE)

print("Predicting probabilities...")

df["probability"] = model.predict_proba(
    df[FEATURES]
)[:, 1]

THRESHOLD = 0.50

pred = df[df["probability"] >= THRESHOLD].copy()

print(f"\nThreshold: {THRESHOLD}")
print(f"Predicted positive pairs: {len(pred):,}")

pred = pred[
    [
        "source1_entity_id",
        "source2_entity_id",
        "probability"
    ]
].sort_values(
    ["source1_entity_id", "probability"],
    ascending=[True, False]
)

pred.to_csv(
    OUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("PREDICTION COMPLETE")
print("=" * 70)

print(f"Output: {OUT_FILE}")
print(f"Predicted pairs: {len(pred):,}")
