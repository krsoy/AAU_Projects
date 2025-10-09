import pandas as pd
import numpy as np
import joblib

# --- Load ---
pipe = joblib.load("final_xgbr_usa_model.pkl")
usa_data = pd.read_csv("usa_salary_data.csv")
usa_25   = pd.read_csv("2025_survey.csv")

LABEL = "CompTotal"

# Reproducible RNG
rng = np.random.default_rng()

def _sample_from_2024(colname, df_2024):
    """Sample a plausible value for one column from the 2024 data distribution."""
    series = df_2024[colname].dropna()
    if series.empty:
        # fallback if column is entirely NaN (unlikely): numeric→0, categorical→most frequent or empty string
        if pd.api.types.is_numeric_dtype(df_2024[colname]):
            return 0
        return ""
    # For both numeric and categorical, sampling from the empirical distribution keeps things realistic
    return series.sample(1, random_state=rng.integers(0, 10_000)).iloc[0]

def build_synthetic_row_with_trace(usa_25, usa_2024, label="CompTotal", rng=None):
    """
    Create one model-ready row by taking a random 2025 row and
    filling any missing 2024-required features with values sampled from 2024.
    Also returns trace of which columns came from which year.
    """
    if rng is None:
        rng = np.random.default_rng()

    expected_features = [c for c in usa_2024.columns if c != label]
    row25 = usa_25.sample(1, random_state=rng.integers(0, 10_000)).iloc[0]

    synthetic = {}
    source_info = {}  # column -> '2025' or '2024'

    for col in expected_features:
        use_25_val = False
        val = None

        if col in row25.index:
            val = row25[col]
            if pd.api.types.is_numeric_dtype(usa_2024[col]):
                val = pd.to_numeric(val, errors="coerce")
            # use 2025 value if not NaN
            if not pd.isna(val):
                use_25_val = True

        if not use_25_val:
            val = _sample_from_2024(col, usa_2024)
            source_info[col] = "2024"
        else:
            source_info[col] = "2025"

        # final type alignment
        if pd.api.types.is_numeric_dtype(usa_2024[col]):
            val = pd.to_numeric(val, errors="coerce")
            if pd.isna(val):
                val = _sample_from_2024(col, usa_2024)
                source_info[col] = "2024"
        else:
            if pd.isna(val):
                val = _sample_from_2024(col, usa_2024)
                source_info[col] = "2024"
            if not isinstance(val, str):
                val = str(val)

        synthetic[col] = val

    X_one = pd.DataFrame([synthetic], columns=expected_features)

    # true label if available
    y_true = None
    if label in row25.index:
        y_true = pd.to_numeric(row25[label], errors="coerce")

    # summary
    used_from_2025 = sum(v == "2025" for v in source_info.values())
    used_from_2024 = sum(v == "2024" for v in source_info.values())
    total = len(source_info)
    pct_2024 = used_from_2024 / total * 100
    pct_2025 = used_from_2025 / total * 100

    print(f"--- Data Completion Report ---")
    print(f"Total model features: {total}")
    print(f"From 2025: {used_from_2025} ({pct_2025:.1f}%)")
    print(f"From 2024: {used_from_2024} ({pct_2024:.1f}%)")

    return X_one, y_true, source_info

# ---- Build one row and predict ----
X_one, y_true, src_info = build_synthetic_row_with_trace(usa_25, usa_data)

y_pred = pipe.predict(X_one)
print("\nPredicted CompTotal:", float(y_pred[0]))
if y_true is not None:
    print("2025 true:", float(y_true))

# Optional: list columns filled from 2024
filled_cols = [col for col, src in src_info.items() if src == "2024"]
print(f"\nColumns filled from 2024 ({len(filled_cols)}):")
print(filled_cols[:20], "..." if len(filled_cols) > 20 else "")
#
# print("Model-ready columns count:", X_one.shape[1])
# print("Predicted CompTotal:", float(y_pred[0]))
# print("2025 true (if present):", None if y_true is None or pd.isna(y_true) else float(y_true))

# --- (Optional) Baseline synthetic purely from 2024, for sanity check ---
def random_sample_row_2024(df_2024, label=LABEL, rng=rng):
    expected_features = [c for c in df_2024.columns if c != label]
    sampled = {}
    for col in expected_features:
        sampled[col] = _sample_from_2024(col, df_2024)
    return pd.DataFrame([sampled], columns=expected_features)

X_2024 = random_sample_row_2024(usa_data, LABEL, rng)
y_pred_2024 = pipe.predict(X_2024)
print("pure 2024 synthetic prediction:", float(y_pred_2024[0]))

