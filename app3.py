# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Dict, Any

st.set_page_config(page_title="USA Salary — Synthetic/Hybrid Prediction", layout="wide")

# --- Top nav to teammates' apps ---
c1, c2, c3 = st.columns(3)
with c1:
    st.link_button("Hamna", "https://example.com/hamna")   # TODO: replace with real link
with c2:
    st.link_button("Mahesh", "https://example.com/mahesh") # TODO: replace with real link
with c3:
    st.link_button("Tian", "https://example.com/tian")     # TODO: replace with real link

st.title("USA Salary — Predict with Hybrid/Synthetic Inputs")

# =============== Load assets ===============
@st.cache_resource(show_spinner=False)
def _load_assets():
    pipe = joblib.load("final_xgbr_usa_model.pkl")
    usa_2024 = pd.read_csv("usa_salary_data.csv")
    usa_2025 = pd.read_csv("2025_survey.csv")
    return pipe, usa_2024, usa_2025

pipe, usa_data, usa_25 = _load_assets()
LABEL = "CompTotal"

# =============== RNG utilities ===============
def _ensure_rng():
    if "rng_seedseq" not in st.session_state:
        st.session_state.rng_seedseq = np.random.SeedSequence()  # fresh entropy
    # Spawn a new child seed each time we need new random numbers
    child = st.session_state.rng_seedseq.spawn(1)[0]
    return np.random.default_rng(child)

def new_rng():
    # Advance the seed sequence so new calls differ
    st.session_state.rng_seedseq = st.session_state.rng_seedseq.spawn(1)[0]
    return np.random.default_rng(st.session_state.rng_seedseq)

# =============== Core sampling functions (adapted from your code) ===============
def _sample_from_2024(colname: str, df_2024: pd.DataFrame, rng: np.random.Generator):
    series = df_2024[colname].dropna()
    if series.empty:
        if pd.api.types.is_numeric_dtype(df_2024[colname]):
            return 0
        return ""
    # sample one from empirical distribution
    # use rng via random_state to keep reproducible within a call
    return series.sample(1, random_state=rng.integers(0, 10_000)).iloc[0]

def build_synthetic_row_with_trace(
    usa_25: pd.DataFrame,
    usa_2024: pd.DataFrame,
    label: str = "CompTotal",
    rng: np.random.Generator | None = None
) -> Tuple[pd.DataFrame, float | None, Dict[str, str], Dict[str, Any]]:
    """
    Returns:
      X_one (DataFrame with expected features),
      y_true (float or None),
      source_info (col -> '2025' or '2024'),
      report (dict with simple counts for UI)
    """
    if rng is None:
        rng = _ensure_rng()

    expected_features = [c for c in usa_2024.columns if c != label]
    row25 = usa_25.sample(1, random_state=rng.integers(0, 10_000)).iloc[0]

    synthetic = {}
    source_info: Dict[str, str] = {}

    for col in expected_features:
        use_25_val = False
        val = None

        if col in row25.index:
            val = row25[col]
            if pd.api.types.is_numeric_dtype(usa_2024[col]):
                val = pd.to_numeric(val, errors="coerce")
            if not pd.isna(val):
                use_25_val = True

        if not use_25_val:
            val = _sample_from_2024(col, usa_2024, rng)
            source_info[col] = "2024"
        else:
            source_info[col] = "2025"

        # final type alignment
        if pd.api.types.is_numeric_dtype(usa_2024[col]):
            val = pd.to_numeric(val, errors="coerce")
            if pd.isna(val):
                val = _sample_from_2024(col, usa_2024, rng)
                source_info[col] = "2024"
        else:
            if pd.isna(val):
                val = _sample_from_2024(col, usa_2024, rng)
                source_info[col] = "2024"
            if not isinstance(val, str):
                val = str(val)

        synthetic[col] = val

    X_one = pd.DataFrame([synthetic], columns=expected_features)

    y_true = None
    if label in row25.index:
        y_true = pd.to_numeric(row25[label], errors="coerce")
        if pd.isna(y_true):
            y_true = None

    used_from_2025 = sum(v == "2025" for v in source_info.values())
    used_from_2024 = sum(v == "2024" for v in source_info.values())
    total = len(source_info)
    report = dict(
        total_features=total,
        n_2025=used_from_2025,
        n_2024=used_from_2024,
        pct_2025=0.0 if total == 0 else used_from_2025 / total * 100,
        pct_2024=0.0 if total == 0 else used_from_2024 / total * 100,
        filled_cols=[c for c, s in source_info.items() if s == "2024"]
    )
    return X_one, y_true, source_info, report

def random_sample_row_2024(
    df_2024: pd.DataFrame,
    label: str,
    rng: np.random.Generator
) -> pd.DataFrame:
    expected_features = [c for c in df_2024.columns if c != label]
    sampled = {}
    for col in expected_features:
        sampled[col] = _sample_from_2024(col, df_2024, rng)
    return pd.DataFrame([sampled], columns=expected_features)

# =============== Session state ===============
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts with prediction, truth, abs_err, pct_err

if "prepared" not in st.session_state:
    st.session_state.prepared = None  # holds dict with keys: X, y_true, info, report, mode

# =============== Sidebar controls ===============
st.sidebar.header("Controls")
mode = st.sidebar.radio(
    "Choose input mode:",
    ["Hybrid (Random 2025 + fill from 2024)", "Pure 2024 synthetic"],
    index=0
)

if st.sidebar.button("Reload new random data"):
    # Force-generate a new candidate without submitting
    rng = new_rng()
    if mode.startswith("Hybrid"):
        X_one, y_true, src_info, rep = build_synthetic_row_with_trace(usa_25, usa_data, label=LABEL, rng=rng)
        st.session_state.prepared = dict(X=X_one, y_true=y_true, info=src_info, report=rep, mode="hybrid")
    else:
        X_2024 = random_sample_row_2024(usa_data, LABEL, rng)
        st.session_state.prepared = dict(X=X_2024, y_true=None, info={}, report={}, mode="pure2024")
    st.toast("New random input prepared.", icon="✅")

# =============== Explanation ===============
with st.expander("What is happening here? (plain English)"):
    st.markdown(
        """
**Goal:** Use your trained model to predict **CompTotal** (total compensation) from one model-ready row.

**Two ways to build that row:**

1) **Hybrid (Random 2025 + fill from 2024)**  
   - We randomly pick a single respondent from the 2025 survey.  
   - Your 2024 model expects a specific set of feature columns. If any of those are **missing in the 2025 row**, we **fill them** by sampling realistic values from the **2024 data distribution** (for that exact column).  
   - This keeps the row consistent with the model’s expected features while using as much fresh 2025 info as possible.

2) **Pure 2024 synthetic**  
   - We **ignore 2025** and build a row by sampling **every feature** directly from the **2024 data distribution**.  
   - This is a sanity check baseline that’s guaranteed to match the model’s training schema.

**Then:**  
- Click **Submit** to predict. If the hybrid row actually includes a true 2025 CompTotal, we’ll show the **prediction vs. truth**, plus **absolute** and **percentage** error.  
- Every submit is added to the **History**, where we compute the **mean deviation** (MAE) and **mean percent error** (MAPE) over all runs with a known truth.  
- After each submit, we **auto-prepare a new random input** so you can predict again immediately.  
- You can also click **Reload new random data** in the sidebar to get a different input before submitting.
        """
    )

st.markdown("---")

# =============== Prepare first candidate if needed ===============
def _prepare_candidate_if_needed():
    if st.session_state.prepared is None:
        rng = _ensure_rng()
        if mode.startswith("Hybrid"):
            X_one, y_true, src_info, rep = build_synthetic_row_with_trace(usa_25, usa_data, label=LABEL, rng=rng)
            st.session_state.prepared = dict(X=X_one, y_true=y_true, info=src_info, report=rep, mode="hybrid")
        else:
            X_2024 = random_sample_row_2024(usa_data, LABEL, rng)
            st.session_state.prepared = dict(X=X_2024, y_true=None, info={}, report={}, mode="pure2024")

_prepare_candidate_if_needed()

# =============== Main predict area ===============
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.subheader("Submit a prediction")

    # Preview current candidate
    if st.session_state.prepared is not None:
        curr = st.session_state.prepared
        mode_tag = "Hybrid (2025+2024)" if curr["mode"] == "hybrid" else "Pure 2024"
        st.caption(f"Next input mode: **{mode_tag}**")

        with st.expander("Show next input row (model-ready features)"):
            st.dataframe(curr["X"].T.rename(columns={0: "value"}))

        if curr["mode"] == "hybrid":
            rep = curr["report"]
            st.caption(
                f"Data completion: **{rep['n_2025']}** from 2025 "
                f"({rep['pct_2025']:.1f}%); **{rep['n_2024']}** from 2024 "
                f"({rep['pct_2024']:.1f}%)"
            )

    # Submit button
    submitted = st.button("Submit & Predict", type="primary", use_container_width=True)

    if submitted and st.session_state.prepared is not None:
        curr = st.session_state.prepared
        X_one = curr["X"]
        y_true = curr["y_true"]

        try:
            y_pred = float(pipe.predict(X_one)[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            y_pred = None

        if y_pred is not None:
            st.success(f"**Predicted CompTotal:** {y_pred:,.0f} USD")

            if y_true is not None:
                st.info(f"**2025 true:** {y_true:,.0f} USD")

                abs_err = abs(y_pred - y_true)
                pct_err = abs_err / y_true * 100 if y_true != 0 else np.nan

                st.write(f"**Absolute error:** {abs_err:,.0f} USD")
                st.write(f"**Percentage error:** {pct_err:.2f}%")

                st.session_state.history.append(
                    dict(pred=y_pred, truth=y_true, abs_err=abs_err, pct_err=pct_err)
                )
            else:
                st.warning("No ground-truth value available for this input (pure 2024 synthetic).")

        # After submit, auto-prepare a new random candidate
        rng = new_rng()
        if mode.startswith("Hybrid"):
            X_one2, y_true2, src_info2, rep2 = build_synthetic_row_with_trace(usa_25, usa_data, label=LABEL, rng=rng)
            st.session_state.prepared = dict(X=X_one2, y_true=y_true2, info=src_info2, report=rep2, mode="hybrid")
        else:
            X_2024b = random_sample_row_2024(usa_data, LABEL, rng)
            st.session_state.prepared = dict(X=X_2024b, y_true=None, info={}, report={}, mode="pure2024")
        st.toast("New random input prepared.", icon="✨")

with right:
    st.subheader("Results history")
    if len(st.session_state.history) == 0:
        st.write("No submissions yet.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df.style.format({"pred": "{:,.0f}", "truth": "{:,.0f}", "abs_err": "{:,.0f}", "pct_err": "{:.2f}"}), use_container_width=True)

        # Mean deviations over entries that have truth
        valid = hist_df.dropna(subset=["truth"])
        if len(valid) > 0:
            mae = valid["abs_err"].mean()
            mape = valid["pct_err"].mean()
            st.metric(label="Mean Absolute Error (USD)", value=f"{mae:,.0f}")
            st.metric(label="Mean Absolute Percentage Error", value=f"{mape:.2f}%")
        else:
            st.write("No entries with ground truth yet.")

st.markdown("---")
st.caption("Tip: If you keep getting the 'same-ish' row, use the sidebar **Reload new random data** to resample.")
