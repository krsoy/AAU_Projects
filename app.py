# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# Config & Links (edit as needed)
# ==============================
st.set_page_config(page_title="2025→2024 Synthetic Salary Predictor", layout="wide")

LINK_HAMNA  = "https://example.com/hamna"   # <-- put real link
LINK_MAHESH = "https://example.com/mahesh"  # <-- put real link
LINK_TIAN   = "https://example.com/tian"    # <-- put real link

MODEL_PATH = "final_xgbr_usa_model.pkl"
DATA_2024_PATH = "usa_salary_data.csv"
DATA_2025_PATH = "2025_survey.csv"
LABEL = "CompTotal"

# ==============================
# Helper: Top Links as Buttons
# ==============================
def link_buttons_row():
    style = """
    <style>
    .linkbar { display:flex; gap:12px; margin-bottom:10px; }
    .linkbtn {
        display:inline-block; padding:8px 14px; border-radius:10px;
        background:#f0f2f6; border:1px solid #dce0e6; text-decoration:none;
        color:#262730; font-weight:600;
    }
    .linkbtn:hover { background:#e6e9ef; }
    </style>
    """
    html = f"""
    <div class="linkbar">
      <a class="linkbtn" href="{LINK_HAMNA}"  target="_blank" rel="noopener">Hamna</a>
      <a class="linkbtn" href="{LINK_MAHESH}" target="_blank" rel="noopener">Mahesh</a>
      <a class="linkbtn" href="{LINK_TIAN}"   target="_blank" rel="noopener">Tian</a>
    </div>
    """
    st.markdown(style + html, unsafe_allow_html=True)

# ==============================
# Load artifacts (cached)
# ==============================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_data():
    usa_data = pd.read_csv(DATA_2024_PATH)
    usa_25   = pd.read_csv(DATA_2025_PATH)
    return usa_data, usa_25

pipe = load_model()
usa_data, usa_25 = load_data()

# RNG
rng = np.random.default_rng()

# ==============================
# Sampling helpers
# ==============================
def _sample_from_2024(colname, df_2024, rng_local=None):
    """Sample a plausible value for one column from 2024 distribution."""
    if rng_local is None: rng_local = rng
    series = df_2024[colname].dropna()
    if series.empty:
        # fallback if column is entirely NaN
        if pd.api.types.is_numeric_dtype(df_2024[colname]):
            return 0
        return ""
    return series.sample(1, random_state=int(rng_local.integers(0, 10_000))).iloc[0]

def build_synthetic_row_with_trace(usa_25_df, usa_2024_df, label=LABEL, rng_local=None):
    """
    Create one model-ready row by taking a random 2025 row and
    filling any missing 2024-required features with values sampled from 2024.
    Also returns a per-column source map: '2025' or '2024'.
    """
    if rng_local is None:
        rng_local = np.random.default_rng()

    # Model input schema = 2024 columns (raw), minus label
    expected_features = [c for c in usa_2024_df.columns if c != label]

    # Sample one 2025 row
    row25 = usa_25_df.sample(1, random_state=int(rng_local.integers(0, 10_000))).iloc[0]

    synthetic = {}
    source_info = {}  # column -> '2025' or '2024'

    for col in expected_features:
        use_25_val = False
        val = None

        if col in row25.index:
            val = row25[col]
            # coerce type to numeric if 2024 expects numeric
            if pd.api.types.is_numeric_dtype(usa_2024_df[col]):
                val = pd.to_numeric(val, errors="coerce")
            if not pd.isna(val):
                use_25_val = True

        if not use_25_val:
            val = _sample_from_2024(col, usa_2024_df, rng_local)
            source_info[col] = "2024"
        else:
            source_info[col] = "2025"

        # Final alignment
        if pd.api.types.is_numeric_dtype(usa_2024_df[col]):
            val = pd.to_numeric(val, errors="coerce")
            if pd.isna(val):
                val = _sample_from_2024(col, usa_2024_df, rng_local)
                source_info[col] = "2024"
        else:
            if pd.isna(val):
                val = _sample_from_2024(col, usa_2024_df, rng_local)
                source_info[col] = "2024"
            if not isinstance(val, str):
                val = str(val)

        synthetic[col] = val

    X_one = pd.DataFrame([synthetic], columns=expected_features)

    # True label if present in 2025 row
    y_true = None
    if label in row25.index:
        y_true = pd.to_numeric(row25[label], errors="coerce")

    # Summary numbers
    used_from_2025 = sum(v == "2025" for v in source_info.values())
    used_from_2024 = sum(v == "2024" for v in source_info.values())
    total = len(source_info)
    return X_one, y_true, source_info, used_from_2025, used_from_2024, total

def random_sample_row_2024(df_2024, label=LABEL, rng_local=None):
    if rng_local is None: rng_local = rng
    expected_features = [c for c in df_2024.columns if c != label]
    sampled = {}
    for col in expected_features:
        sampled[col] = _sample_from_2024(col, df_2024, rng_local)
        # type alignment:
        if pd.api.types.is_numeric_dtype(df_2024[col]):
            sampled[col] = pd.to_numeric(sampled[col], errors="coerce")
            if pd.isna(sampled[col]):
                sampled[col] = 0
        else:
            if sampled[col] is None or (isinstance(sampled[col], float) and pd.isna(sampled[col])):
                sampled[col] = ""
            sampled[col] = str(sampled[col])
    return pd.DataFrame([sampled], columns=expected_features)

# ==============================
# Session state for history & prepared sample
# ==============================
if "prepared_X" not in st.session_state:
    st.session_state.prepared_X = None
if "prepared_ytrue" not in st.session_state:
    st.session_state.prepared_ytrue = None
if "source_map" not in st.session_state:
    st.session_state.source_map = None
if "history" not in st.session_state:
    # list of dicts: {pred, ytrue, abs_err, pct_err}
    st.session_state.history = []

# ==============================
# Top Links + Explanation
# ==============================
link_buttons_row()

st.markdown(
    """
### How this works (plain English)
We **take one respondent from 2025** and use any matching columns that the 2024 model needs.  
If a 2024 feature is **missing or incompatible** in the 2025 row, we **fill it** by randomly sampling from the **2024 data distribution** (so values stay realistic).  
This creates a **complete input row** that the 2024-trained model can accept.

**What you do:**
1. Pick a source: **“Random 2025 + 2024 fill”** or **“Pure 2024 synthetic”**.  
2. Click **Generate sample**.  
3. Optionally **edit any feature** (toggle “Use prepared value?” → choose a category or type a number).  
4. Click **Submit for prediction** to see **Predicted vs True** (if available), plus errors.  
5. We keep a running **history**, so you can see average deviation.
"""
)

# ==============================
# Sidebar: Controls
# ==============================
st.sidebar.header("Controls")

mode = st.sidebar.radio(
    "Choose data source:",
    ["Random 2025 + 2024 fill", "Pure 2024 synthetic"],
)

seed = st.sidebar.number_input(
    "Random seed (optional)", min_value=0, max_value=10_000_000, value=0, step=1
)
use_seed = st.sidebar.checkbox("Use the seed above", value=False)

def make_rng():
    if use_seed:
        return np.random.default_rng(int(seed))
    return np.random.default_rng()

colA, colB = st.sidebar.columns(2)
with colA:
    gen_clicked = st.button("Generate sample", type="primary", use_container_width=True)
with colB:
    reset_hist = st.button("Reset history", use_container_width=True)

if reset_hist:
    st.session_state.history = []
    st.success("History cleared.")

# ==============================
# Generate sample per mode
# ==============================
if gen_clicked or (st.session_state.prepared_X is None):
    local_rng = make_rng()
    if mode == "Random 2025 + 2024 fill":
        X_one, y_true, source_map, n25, n24, total = build_synthetic_row_with_trace(
            usa_25, usa_data, LABEL, rng_local=local_rng
        )
        st.session_state.prepared_X = X_one
        st.session_state.prepared_ytrue = y_true
        st.session_state.source_map = source_map
        st.session_state.data_completion = (n25, n24, total)
    else:
        X_2024 = random_sample_row_2024(usa_data, LABEL, rng_local=local_rng)
        st.session_state.prepared_X = X_2024
        st.session_state.prepared_ytrue = np.nan  # no true label for pure 2024
        st.session_state.source_map = {c: "2024" for c in X_2024.columns}
        st.session_state.data_completion = (0, X_2024.shape[1], X_2024.shape[1])

# ==============================
# Show Data Completion Report
# ==============================
if st.session_state.prepared_X is not None:
    n25, n24, total = st.session_state.data_completion
    if total > 0:
        pct25 = 100 * n25 / total
        pct24 = 100 * n24 / total
    else:
        pct25 = pct24 = 0.0

    st.subheader("Data Completion Report")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total features", total)
    c2.metric("From 2025", f"{n25} ({pct25:.1f}%)")
    c3.metric("From 2024", f"{n24} ({pct24:.1f}%)")

# ==============================
# Editor: Per-feature controls
# ==============================
st.subheader("Feature Editor")

X_prepared = st.session_state.prepared_X.copy()
source_map = st.session_state.source_map or {}
y_true = st.session_state.prepared_ytrue

# Build per-column editor
edited_vals = {}
use_prepared = {}

# For categorical choices, pull from 2024 distinct values (for consistency)
value_options = {
    col: sorted(pd.Series(usa_data[col].dropna().astype(str)).unique().tolist())[:2000]
    for col in X_prepared.columns
    if not pd.api.types.is_numeric_dtype(usa_data[col])
}

with st.form("edit_form"):
    # Chunk the feature list to avoid too-long pages; but keep all editable
    for col in X_prepared.columns:
        cur_val = X_prepared.iloc[0][col]
        dtype_is_num = pd.api.types.is_numeric_dtype(usa_data[col])

        # Layout per row
        row = st.container()
        with row:
            c1, c2, c3 = st.columns([2.2, 1.2, 3.0])
            c1.write(f"**{col}**  \nSource: `{source_map.get(col, 'N/A')}`")

            # toggle: use prepared?
            default_use = True
            use_flag = c2.checkbox(
                "Use prepared?",
                value=default_use,
                key=f"use_{col}"
            )
            use_prepared[col] = use_flag

            # editor
            if use_flag:
                # show the prepared value (read-only)
                c3.write(f"`{cur_val}`")
                edited_vals[col] = cur_val
            else:
                if dtype_is_num:
                    # numeric input
                    # try to set a reasonable default float
                    try:
                        default_num = float(cur_val) if pd.notna(cur_val) else 0.0
                    except Exception:
                        default_num = 0.0
                    new_val = c3.number_input(
                        "Numeric value",
                        value=float(default_num),
                        key=f"num_{col}"
                    )
                    edited_vals[col] = new_val
                else:
                    # categorical input (select from 2024 options, include current if missing)
                    opts = value_options.get(col, [])
                    cur_str = str(cur_val)
                    if cur_str not in opts:
                        opts = [cur_str] + opts
                    new_val = c3.selectbox(
                        "Choose category",
                        options=opts if len(opts) > 0 else [cur_str],
                        index=0,
                        key=f"cat_{col}"
                    )
                    edited_vals[col] = new_val

    submitted = st.form_submit_button("Submit for prediction", type="primary")

# ==============================
# Prediction + History
# ==============================
def coerce_types_like_2024(df_row: pd.DataFrame, df_2024: pd.DataFrame):
    """Coerce column types to match 2024 schema."""
    fixed = {}
    for col in df_row.columns:
        val = df_row.iloc[0][col]
        if pd.api.types.is_numeric_dtype(df_2024[col]):
            val = pd.to_numeric(val, errors="coerce")
            if pd.isna(val): val = 0
        else:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                val = ""
            val = str(val)
        fixed[col] = val
    return pd.DataFrame([fixed], columns=df_row.columns)

if submitted:
    # Construct the final input row with edits
    final_row = pd.DataFrame([edited_vals], columns=X_prepared.columns)
    final_row = coerce_types_like_2024(final_row, usa_data)

    # Predict
    # NOTE: The pipeline should internally preprocess (OHE with handle_unknown="ignore", scales, etc.)
    y_pred = float(pipe.predict(final_row)[0])

    # True & errors
    has_true = y_true is not None and pd.notna(y_true)
    if has_true:
        true_val = float(y_true)
        abs_err = abs(y_pred - true_val)
        pct_err = abs_err / true_val * 100 if true_val != 0 else np.nan
    else:
        true_val = np.nan
        abs_err = np.nan
        pct_err = np.nan

    # Show results
    st.subheader("Prediction Result")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Predicted CompTotal", f"{y_pred:,.2f}")
    r2.metric("True (if available)", "—" if not has_true else f"{true_val:,.2f}")
    r3.metric("Absolute Error", "—" if not has_true else f"{abs_err:,.2f}")
    r4.metric("Percent Error", "—" if not has_true or np.isnan(pct_err) else f"{pct_err:.2f}%")

    # Record to history if we have a true value
    st.session_state.history.append({
        "pred": y_pred,
        "ytrue": true_val,
        "abs_err": abs_err,
        "pct_err": pct_err
    })

# ==============================
# History & Summary
# ==============================
st.subheader("Evaluation History")

if len(st.session_state.history) == 0:
    st.info("No submissions yet. Generate a sample, make any edits you want, and click **Submit for prediction**.")
else:
    hist_df = pd.DataFrame(st.session_state.history)
    # Show table
    st.dataframe(
        hist_df.style.format({
            "pred": "{:,.2f}",
            "ytrue": "{:,.2f}",
            "abs_err": "{:,.2f}",
            "pct_err": "{:.2f}"
        }),
        use_container_width=True
    )
    # Summary metrics (consider only rows with valid true)
    valid = hist_df.dropna(subset=["ytrue"])
    if len(valid) > 0:
        mad = valid["abs_err"].mean()
        mpe = valid["pct_err"].mean()
        s1, s2 = st.columns(2)
        s1.metric("Mean Absolute Deviation", f"{mad:,.2f}")
        s2.metric("Mean Percent Error", f"{mpe:.2f}%")
    else:
        st.info("No rows with a true value yet (pure 2024 synthetic has no ground truth).")

# ==============================
# Optional: Show prepared row & sources
# ==============================
with st.expander("See prepared row and per-feature source"):
    st.write("**Prepared (editable) row:**")
    st.dataframe(st.session_state.prepared_X, use_container_width=True)
    # Source map table
    src_df = pd.DataFrame({
        "Feature": list(st.session_state.source_map.keys()),
        "Source":  list(st.session_state.source_map.values())
    })
    st.write("**Feature source (2024 vs 2025):**")
    st.dataframe(src_df, use_container_width=True)
