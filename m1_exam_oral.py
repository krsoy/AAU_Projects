# Step 1: Overview 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------- Page setup ----------
st.set_page_config(page_title="Developer Pay & Satisfaction", page_icon="💼", layout="wide")

# Small brand header
st.markdown(
    """
    <h2 style="margin-bottom:0">💼 Developer Insights Dashboard</h2>
    <p style="color:#6c757d;margin-top:2px">Explore salary & satisfaction across countries — commercial-grade, interactive & fast.</p>
    <hr style="margin:8px 0 16px 0" />
    """,
    unsafe_allow_html=True,
)

# ---------- Data loader ----------
# ---------- Data loader (the ONLY source of truth) ----------
import os
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=True)
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "survey_results_public.csv")

    df = pd.read_csv(data_path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Detect salary column
    salary_candidates = ["ConvertedCompYearly", "CompTotal", "ConvertedCompYearly "]
    salary_col = next((c for c in salary_candidates if c in df.columns), None)

    if salary_col is not None:
        df[salary_col] = pd.to_numeric(df[salary_col], errors="coerce")

    if "YearsCodePro" in df.columns:
        df["YearsCodePro"] = (
            df["YearsCodePro"]
            .replace({"Less than 1 year": 0, "More than 50 years": 51})
        )
        df["YearsCodePro"] = pd.to_numeric(df["YearsCodePro"], errors="coerce")

    if "OrgSize" in df.columns:
        df["OrgSize"] = df["OrgSize"].astype("category")

    if "JobSat" in df.columns:
        tmp = pd.to_numeric(df["JobSat"], errors="coerce")
        if tmp.isna().all():
            tmp = (
                df["JobSat"].astype(str)
                .str.extract(r"(\d+\.?\d*)")[0]
                .astype(float)
            )
        df["JobSat_num"] = tmp

    return df, salary_col

# Call it ONCE
df, SAL = load_data()

# Debug (临时排错用，用完可删)
st.caption(f"Loaded file in: {os.path.dirname(os.path.abspath(__file__))}")
st.write("First 12 columns:", df.columns[:12].tolist())
st.write("Detected salary column:", SAL)

if SAL is None:
    # 进一步提示可能的候选列，帮助你确认是否列名不同
    cand = [c for c in df.columns if any(k in c.lower() for k in ["comp", "salary", "pay"])]
    st.warning(f"Comp-related columns I can see: {cand[:20]}")
    st.error("No salary column found (ConvertedCompYearly / CompTotal missing). Please check your dataset.")
    st.stop()
    # Years of professional coding
    if "YearsCodePro" in df.columns:
        # handle 'Less than 1 year', 'More than 50 years'
        df["YearsCodePro"] = (
            df["YearsCodePro"]
            .replace({"Less than 1 year": 0, "More than 50 years": 51})
        )
        df["YearsCodePro"] = pd.to_numeric(df["YearsCodePro"], errors="coerce")

    # Org size is categorical; keep as-is if present
    if "OrgSize" in df.columns:
        df["OrgSize"] = df["OrgSize"].astype("category")

    # Job satisfaction — try to make it numeric 1-10
    if "JobSat" in df.columns:
        df["JobSat_num"] = pd.to_numeric(df["JobSat"], errors="coerce")
        # if still empty, try to parse like '8.0' embedded in strings
        if df["JobSat_num"].isna().all():
            df["JobSat_num"] = (
                df["JobSat"]
                .astype(str)
                .str.extract(r"(\d+\.?\d*)")
                .astype(float)
                .squeeze()
            )

    return df, salary_col

df, SAL = load_data()
if SAL is None:
    st.error("No salary column found (ConvertedCompYearly / CompTotal missing). Please check your dataset.")
    st.stop()

# ---------- Sidebar filters ----------
with st.sidebar:
    st.header("Filters")
    # Country filter (top-by-count as default)
    if "Country" in df.columns:
        top_countries = df["Country"].value_counts().head(15).index.tolist()
        countries = st.multiselect(
            "Country", options=sorted(df["Country"].dropna().unique()),
            default=[c for c in ["United States of America",
                                 "United Kingdom of Great Britain and Northern Ireland",
                                 "Germany", "France", "Canada"] if c in df["Country"].unique()]
                     or top_countries[:5]
        )
    else:
        countries = []

    # Years of professional coding
    if "YearsCodePro" in df.columns and df["YearsCodePro"].notna().any():
        min_yrs, max_yrs = int(df["YearsCodePro"].min()), int(df["YearsCodePro"].max())
        yrs_range = st.slider("Years of professional coding", min_yrs, max_yrs, (min_yrs, max_yrs))
    else:
        yrs_range = None

    # Org size
    if "OrgSize" in df.columns:
        org_choices = ["All"] + df["OrgSize"].dropna().unique().tolist()
        org_pick = st.selectbox("Organization size", options=org_choices, index=0)
    else:
        org_pick = "All"

# ---------- Apply filters ----------
df_f = df.copy()

if countries:
    df_f = df_f[df_f["Country"].isin(countries)]

if yrs_range and "YearsCodePro" in df_f.columns:
    df_f = df_f[df_f["YearsCodePro"].between(yrs_range[0], yrs_range[1], inclusive="both")]

if org_pick != "All" and "OrgSize" in df_f.columns:
    df_f = df_f[df_f["OrgSize"] == org_pick]

# Remove obviously invalid salaries
df_f = df_f[df_f[SAL].between(1_000, 300_000, inclusive="both")]

# ---------- KPI row ----------
c1, c2, c3, c4 = st.columns(4)
sample_n = int(df_f[SAL].dropna().shape[0])
median_salary = df_f[SAL].median()
p90_salary = df_f[SAL].quantile(0.90)
avg_job_sat = df_f.get("JobSat_num", pd.Series(dtype=float)).mean()

c1.metric("Sample size", f"{sample_n:,}")
c2.metric("Median salary (USD)", f"${median_salary:,.0f}")
c3.metric("P90 salary (USD)", f"${p90_salary:,.0f}")
c4.metric("Avg. Job Satisfaction", f"{avg_job_sat:.2f}" if not np.isnan(avg_job_sat) else "—")

st.caption("Filters apply to all metrics and charts below.")

# ---------- Row 1: Salary distribution ----------
left, right = st.columns([1.15, 1])
with left:
    st.subheader("Salary distribution")
    fig = px.histogram(
        df_f, x=SAL, nbins=50,
        color="Country" if countries and len(countries) > 1 and "Country" in df_f.columns else None,
        opacity=0.85, template="plotly_white"
    )
    fig.update_layout(
        xaxis_title="Annual salary (USD)",
        yaxis_title="Count",
        bargap=0.05,
        legend_title="Country"
    )
    fig.update_xaxes(range=[0, 300_000])
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Median salary by country")
    if "Country" in df_f.columns and not df_f.empty:
        med = (df_f[[SAL, "Country"]]
               .dropna()
               .groupby("Country")[SAL]
               .median()
               .sort_values(ascending=False)
               .head(12)
               .reset_index())
        fig2 = px.bar(
            med, x=SAL, y="Country", orientation="h",
            text=SAL, template="plotly_white", color="Country", color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig2.update_layout(
            xaxis_title="Median salary (USD)",
            yaxis_title="",
            showlegend=False
        )
        fig2.update_traces(texttemplate="$%{x:,.0f}", textposition="outside", cliponaxis=False)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Country column missing or no data after filters.")

# ---------- Row 2: Salary vs Experience (binned box) ----------
st.subheader("Salary vs experience (binned)")
if "YearsCodePro" in df_f.columns and df_f["YearsCodePro"].notna().any():
    # Create commercial-friendly bins
    bins = [-1, 2, 5, 10, 15, 20, 100]
    labels = ["0-2", "3-5", "6-10", "11-15", "16-20", "20+"]
    tmp = df_f[[SAL, "YearsCodePro"]].dropna().copy()
    tmp["ExpBin"] = pd.cut(tmp["YearsCodePro"], bins=bins, labels=labels)
    fig3 = px.box(
        tmp, x="ExpBin", y=SAL, template="plotly_white",
        color="ExpBin", category_orders={"ExpBin": labels}
    )
    fig3.update_layout(xaxis_title="Years of professional coding (binned)", yaxis_title="Salary (USD)", showlegend=False)
    fig3.update_yaxes(range=[0, 300_000])
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("YearsCodePro column missing or empty.")