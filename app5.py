# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Dict, Any, List

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

# =================== Load assets ===================
@st.cache_resource(show_spinner=False)
def _load_assets():
    pipe = joblib.load("final_xgbr_usa_model.pkl")
    usa_2024 = pd.read_csv("usa_salary_data.csv")
    usa_2025 = pd.read_csv("2025_survey.csv")
    return pipe, usa_2024, usa_2025

pipe, usa_data, usa_25 = _load_assets()
LABEL = "CompTotal"

# =================== RNG utilities ===================
def _ensure_rng():
    if "rng_seedseq" not in st.session_state:
        st.session_state.rng_seedseq = np.random.SeedSequence()
    child = st.session_state.rng_seedseq.spawn(1)[0]
    return np.random.default_rng(child)

def new_rng():
    st.session_state.rng_seedseq = st.session_state.rng_seedseq.spawn(1)[0]
    return np.random.default_rng(st.session_state.rng_seedseq)

# =================== Precompute dropdown choices (ONCE) ===================
@st.cache_resource(show_spinner=False)
def _precompute_choices(usa_2024: pd.DataFrame, usa_2025: pd.DataFrame, label: str) -> Dict[str, List]:
    CHOICES: Dict[str, List] = {}
    both = pd.concat([usa_2024, usa_2025.reindex(columns=usa_2024.columns, fill_value=np.nan)],
                     axis=0, ignore_index=True)
    for col in usa_2024.columns:
        if col == label:
            continue
        s = both[col]
        if pd.api.types.is_numeric_dtype(usa_2024[col]):
            s_num = pd.to_numeric(s, errors="coerce").dropna()
            if len(s_num) == 0:
                CHOICES[col] = [0]
                continue
            q_list = [5,10,20,30,40,50,60,70,80,90,95]
            qs = np.percentile(s_num, q_list).tolist()
            rounded = (np.round(s_num / 1000) * 1000).astype(int)
            top_round = rounded.value_counts().head(10).index.astype(int).tolist()
            merged = sorted(set(int(v) for v in qs + top_round))
            CHOICES[col] = merged[:50]
        else:
            s_cat = s.astype(str).replace({"nan": None}).dropna()
            CHOICES[col] = (s_cat.value_counts().head(40).index.tolist()
                            if len(s_cat) else [""])
    return CHOICES

CHOICES_DICT = _precompute_choices(usa_data, usa_25, LABEL)

# =================== Core sampling funcs ===================
def _sample_from_2024(colname: str, df_2024: pd.DataFrame, rng: np.random.Generator):
    series = df_2024[colname].dropna()
    if series.empty:
        return 0 if pd.api.types.is_numeric_dtype(df_2024[colname]) else ""
    return series.sample(1, random_state=rng.integers(0, 10_000)).iloc[0]

def build_synthetic_row_with_trace(
    usa_25: pd.DataFrame,
    usa_2024: pd.DataFrame,
    label: str = "CompTotal",
    rng: np.random.Generator | None = None
) -> Tuple[pd.DataFrame, float | None, Dict[str, str], Dict[str, Any]]:
    if rng is None:
        rng = _ensure_rng()
    expected_features = [c for c in usa_2024.columns if c != label]
    row25 = usa_25.sample(1, random_state=rng.integers(0, 10_000)).iloc[0]

    synthetic, source_info = {}, {}
    for col in expected_features:
        use_25_val, val = False, None
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

        if pd.api.types.is_numeric_dtype(usa_2024[col]):
            val = pd.to_numeric(val, errors="coerce")
            if pd.isna(val):
                val = _sample_from_2024(col, usa_2024, rng); source_info[col] = "2024"
        else:
            if pd.isna(val):
                val = _sample_from_2024(col, usa_2024, rng); source_info[col] = "2024"
            if not isinstance(val, str):
                val = str(val)
        synthetic[col] = val

    X_one = pd.DataFrame([synthetic], columns=expected_features)

    y_true = None
    if label in row25.index:
        y_true = pd.to_numeric(row25[label], errors="coerce")
        if pd.isna(y_true): y_true = None

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

def random_sample_row_2024(df_2024: pd.DataFrame, label: str, rng: np.random.Generator) -> pd.DataFrame:
    expected_features = [c for c in df_2024.columns if c != label]
    sampled = {col: _sample_from_2024(col, df_2024, rng) for col in expected_features}
    return pd.DataFrame([sampled], columns=expected_features)

# =================== Session state ===================
if "history" not in st.session_state:
    st.session_state.history = []
if "prepared" not in st.session_state:
    st.session_state.prepared = None
# version token to force editor widgets to refresh when a new candidate is created
if "edit_version" not in st.session_state:
    st.session_state.edit_version = 0

def _bump_edit_version():
    st.session_state.edit_version += 1

# =================== Sidebar ===================
st.sidebar.header("Controls")
mode = st.sidebar.radio(
    "Choose input mode:",
    ["Hybrid (Random 2025 + fill from 2024)", "Pure 2024 synthetic"],
    index=0
)

if st.sidebar.button("Reload new random data"):
    rng = new_rng()
    if mode.startswith("Hybrid"):
        X_one, y_true, src_info, rep = build_synthetic_row_with_trace(usa_25, usa_data, label=LABEL, rng=rng)
        st.session_state.prepared = dict(X=X_one, y_true=y_true, info=src_info, report=rep, mode="hybrid")
    else:
        X_2024 = random_sample_row_2024(usa_data, LABEL, rng)
        st.session_state.prepared = dict(X=X_2024, y_true=None, info={}, report={}, mode="pure2024")
    _bump_edit_version()
    st.toast("New random input prepared.", icon="✅")

# =================== Explanation ===================
with st.expander("What is happening here? (plain English)"):
    st.markdown(
        """
**Goal:** USA coder annual income prediction.

**Modes:**
1) **Hybrid** — Random 2025 row; missing required features are filled with samples from the 2024 distribution.
2) **Pure 2024** — Entire row sampled from the 2024 distribution.

**Editing flow:**  
Use the editors below to adjust values. Then click **Apply edits** (this commits your selections to the current row).  
Finally click **Submit & Predict**.
        """
    )

st.markdown("---")

# =================== Prepare first candidate if needed ===================
def _prepare_candidate_if_needed():
    if st.session_state.prepared is None:
        rng = _ensure_rng()
        if mode.startswith("Hybrid"):
            X_one, y_true, src_info, rep = build_synthetic_row_with_trace(usa_25, usa_data, label=LABEL, rng=rng)
            st.session_state.prepared = dict(X=X_one, y_true=y_true, info=src_info, report=rep, mode="hybrid")
        else:
            X_2024 = random_sample_row_2024(usa_data, LABEL, rng)
            st.session_state.prepared = dict(X=X_2024, y_true=None, info={}, report={}, mode="pure2024")
        _bump_edit_version()

_prepare_candidate_if_needed()

# =================== Helper: coerce edited values ===================
def _coerce_value(col: str, val):
    if pd.api.types.is_numeric_dtype(usa_data[col]):
        try:
            return pd.to_numeric(val)
        except Exception:
            return np.nan
    else:
        return "" if val is None else str(val)
# =================== Predict & History ===================
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.subheader("Submit a prediction")

    curr = st.session_state.prepared
    mode_tag = "Hybrid (2025+2024)" if curr["mode"] == "hybrid" else "Pure 2024"
    st.caption(f"Next input mode: **{mode_tag}**")

    with st.expander("Show current input row (after Apply)", expanded=False):
        st.dataframe(curr["X"].T.rename(columns={0: "value"}))

    if curr["mode"] == "hybrid":
        rep = curr["report"]
        st.caption(
            f"Data completion: **{rep['n_2025']}** from 2025 "
            f"({rep['pct_2025']:.1f}%); **{rep['n_2024']}** from 2024 "
            f"({rep['pct_2024']:.1f}%)"
        )

    submitted = st.button("Submit & Predict", type="primary", use_container_width=True)

    if submitted:
        X_one = curr["X"].copy()
        y_true = curr["y_true"]

        # dtype alignment
        for col in X_one.columns:
            if pd.api.types.is_numeric_dtype(usa_data[col]):
                X_one[col] = pd.to_numeric(X_one[col], errors="coerce")
            else:
                X_one[col] = X_one[col].astype(str).fillna("")

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

        # auto-prepare a new candidate after submit
        rng = new_rng()
        if mode.startswith("Hybrid"):
            X_one2, y_true2, src_info2, rep2 = build_synthetic_row_with_trace(usa_25, usa_data, label=LABEL, rng=rng)
            st.session_state.prepared = dict(X=X_one2, y_true=y_true2, info=src_info2, report=rep2, mode="hybrid")
        else:
            X_2024b = random_sample_row_2024(usa_data, LABEL, rng)
            st.session_state.prepared = dict(X=X_2024b, y_true=None, info={}, report={}, mode="pure2024")
        _bump_edit_version()  # refresh editor defaults for the new candidate
        st.toast("New random input prepared.", icon="✨")

with right:
    st.subheader("Results history")
    if len(st.session_state.history) == 0:
        st.write("No submissions yet.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(
            hist_df.style.format({"pred": "{:,.0f}", "truth": "{:,.0f}", "abs_err": "{:,.0f}", "pct_err": "{:.2f}"}),
            use_container_width=True
        )
        valid = hist_df.dropna(subset=["truth"])
        if len(valid) > 0:
            mae = valid["abs_err"].mean()
            mape = valid["pct_err"].mean()
            st.metric(label="Mean Absolute Error (USD)", value=f"{mae:,.0f}")
            st.metric(label="Mean Absolute Percentage Error", value=f"{mape:.2f}%")
        else:
            st.write("No entries with ground truth yet.")

st.markdown("---")
st.caption("Workflow: Edit → **Apply edits** → **Submit & Predict**. Use the sidebar to reload a fresh random row.")
# =================== Editable input row (FORM + Apply button) ===================
st.subheader("Edit current input (optional)")

if st.session_state.prepared is not None:
    curr = st.session_state.prepared
    X_row = curr["X"].iloc[0]  # view (we’ll copy when applying)
    version = st.session_state.edit_version

    with st.form(key=f"edit_form_{version}", clear_on_submit=False):

        applied = st.form_submit_button("Apply edits", use_container_width=True,type="primary")
        exp1 = st.expander("Categorical features", expanded=True)
        exp2 = st.expander("Numeric features", expanded=False)

        cat_cols = [c for c in X_row.index if not pd.api.types.is_numeric_dtype(usa_data[c])]
        num_cols = [c for c in X_row.index if pd.api.types.is_numeric_dtype(usa_data[c])]

        with exp1:
            st.caption("Pick from common categories (precomputed).")
            for col in cat_cols:
                choices = CHOICES_DICT.get(col, [])
                curr_val = "" if pd.isna(X_row[col]) else str(X_row[col])
                if curr_val not in choices and curr_val != "":
                    choices = [curr_val] + choices
                sel = st.selectbox(
                    label=col,
                    options=choices if len(choices) > 0 else [""],
                    index=(choices.index(curr_val) if curr_val in choices and len(choices)>0 else 0),
                    key=f"edit_cat_{col}_{version}",
                )

        with exp2:
            st.caption("Pick typical numeric values (percentiles/rounded).")
            for col in num_cols:
                choices = CHOICES_DICT.get(col, [])
                curr_val = X_row[col]
                if pd.isna(curr_val):
                    curr_val = choices[0] if len(choices) else 0
                if len(choices) == 0:
                    choices = [curr_val]
                elif curr_val not in choices:
                    choices = [curr_val] + choices
                sel = st.selectbox(
                    label=col,
                    options=choices,
                    index=(choices.index(curr_val) if curr_val in choices else 0),
                    key=f"edit_num_{col}_{version}",
                )

    if applied:
        # Build a fresh row dict from widget states
        new_row = {}
        for col in X_row.index:
            key = (f"edit_cat_{col}_{version}"
                   if not pd.api.types.is_numeric_dtype(usa_data[col])
                   else f"edit_num_{col}_{version}")
            if key in st.session_state:
                new_row[col] = _coerce_value(col, st.session_state[key])
            else:
                new_row[col] = X_row[col]

        # IMPORTANT: replace the whole DF to avoid chained-assignment pitfalls
        new_df = pd.DataFrame([new_row], columns=curr["X"].columns)

        # Commit atomically (don’t mutate nested objects in-place)
        st.session_state.prepared = {
            **st.session_state.prepared,
            "X": new_df
        }

        st.toast("Edits applied to current input.", icon="✍️")
        st.rerun()  # ensure the preview below reflects the new DF immediately


