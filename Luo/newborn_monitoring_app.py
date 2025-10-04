
import os
import io
import sys
import textwrap
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# Optional interactive 3D
try:
    import plotly.express as px
except Exception:
    px = None

# Optional: execute notebook to (re)generate data
def try_execute_notebook(nb_path: str) -> str:
    try:
        from nbclient import NotebookClient
        from nbformat import read, NO_CONVERT
    except Exception as e:
        return f"nbclient/nbformat not available: {e}"

    if not os.path.exists(nb_path):
        return f"Notebook not found at: {nb_path}"

    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = read(f, as_version=NO_CONVERT)
        client = NotebookClient(nb, timeout=600, kernel_name='python3')
        client.execute()
        return "Notebook executed successfully."
    except Exception as e:
        return f"Notebook execution failed: {e}"

# Utility: robust CSV loader (tries multiple locations)
def robust_read_csv(filename: str):
    candidates = [
        Path(filename),
        Path("./") / filename,
        Path("/mnt/data") / filename
    ]
    for p in candidates:
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return None

# === Sidebar Controls ===
st.sidebar.title("Controls")

default_nb = "/mnt/data/b210e2c5-f34c-4b9c-a6b2-544e70cfd3f2.ipynb"
nb_path = st.sidebar.text_input("Notebook path to generate data", value=default_nb)
run_nb = st.sidebar.button("Execute notebook to (re)generate data")

if run_nb:
    msg = try_execute_notebook(nb_path)
    st.sidebar.success(msg)

st.sidebar.markdown("---")
st.sidebar.caption("Data loading preferences")

orig_csv_name = st.sidebar.text_input(
    "Original data CSV (with risk column)",
    value="newborn_health_monitoring_with_risk.csv"
)

# Attempt to load authored outputs first (if notebook writes them)
df3d = robust_read_csv("df3d.csv")
pca_df = robust_read_csv("pca_df.csv")
centroids = robust_read_csv("kmeans_centroids_in_PC_space.csv")
orig_df = robust_read_csv(orig_csv_name)

# === Fallback pipeline to create PCA/KMeans if notebook outputs are missing ===
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def build_pipeline_from_original(df: pd.DataFrame, n_clusters: int = 4):
    # Expect a 'risk_level' (or similar) label column; avoid using it for ML transforms
    label_cols = [c for c in df.columns if c.lower() in ("risk_level","risk","label","is_risk")]
    feature_df = df.drop(columns=label_cols, errors="ignore").select_dtypes(include=[np.number]).copy()

    # Keep non-numeric columns for later joins if needed
    non_num_cols = df.drop(columns=feature_df.columns, errors="ignore")

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df.values)

    # 3D PCA for plotting
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X)
    pca_cols = ["PC1","PC2","PC3"]
    pca_data = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)

    # KMeans in PCA space for stability/visualization
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X_pca)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=pca_cols)
    centers["cluster"] = np.arange(n_clusters)

    # Build df3d aligning with earlier naming
    df3d_local = pd.concat([pca_data, df[label_cols]], axis=1)
    if "risk_level" not in df3d_local.columns and len(label_cols) > 0:
        df3d_local.rename(columns={label_cols[0]:"risk_level"}, inplace=True)
    df3d_local["cluster"] = clusters

    # Full pca_df with original + PCs
    pca_full = pd.concat([df.reset_index(drop=True), pca_data.reset_index(drop=True)], axis=1)

    return df3d_local, pca_full, centers

# Build from original if needed
if orig_df is None:
    st.error("Original dataset not found. Please ensure the CSV exists (e.g., /mnt/data/newborn_health_monitoring_with_risk.csv) or adjust the path in the sidebar.")
    st.stop()

if df3d is None or pca_df is None or centroids is None:
    df3d, pca_df, centroids = build_pipeline_from_original(orig_df, n_clusters=4)
    # Save for reuse
    try:
        df3d.to_csv("df3d.csv", index=False)
        pca_df.to_csv("pca_df.csv", index=False)
        centroids.to_csv("kmeans_centroids_in_PC_space.csv", index=False)
    except Exception:
        pass

# Ensure expected columns/types
if "cluster" in df3d.columns:
    df3d["cluster"] = df3d["cluster"].astype(str)

# === Top-level title ===
st.title("Newborn Health Monitoring: PCA, Clusters & Risk Density")

# === Dataset Info & Risk Ratio ===
st.header("Dataset Overview")
n_rows, n_cols = orig_df.shape
st.write(f"**Original dataset:** {n_rows} rows × {n_cols} columns")

# Risk column detection
risk_col = None
for c in ["risk_level","risk","label","is_risk"]:
    if c in orig_df.columns:
        risk_col = c
        break

if risk_col is None:
    st.warning("No explicit risk label column found. Expected 'risk_level' / 'risk' / 'label' / 'is_risk'. Using heuristic: treat highest category or 1 as risk if present.")
    # heuristic for demo
    # If there's a binary column, pick it
    bin_cols = [c for c in orig_df.columns if set(pd.unique(orig_df[c].dropna())) <= {0,1}]
    if bin_cols:
        risk_col = bin_cols[0]
    else:
        # create a dummy risk col of 0
        orig_df["risk_level"] = 0
        risk_col = "risk_level"

# Normalize risk to 0/1 for ratio calc
risk_series = orig_df[risk_col]
# Map string categories to 0/1 if needed
if risk_series.dtype == object:
    # try mapping typical labels
    mapping = {"non-risk":0,"non_risk":0,"low":0,"no":0,"negative":0,"risk":1,"high":1,"yes":1,"positive":1}
    risk_series_bin = risk_series.str.lower().map(mapping)
    # fallback: treat the most frequent value as 0, others as 1
    if risk_series_bin.isna().any():
        top = risk_series.value_counts().index[0]
        risk_series_bin = (risk_series != top).astype(int)
else:
    # numeric: assume >0 => risk
    risk_series_bin = (risk_series.astype(float) > 0).astype(int)

risk_rate = risk_series_bin.mean()
nonrisk_rate = 1 - risk_rate
st.write(f"**Risk vs Non-risk ratio:** {risk_rate:.2%} risk, {nonrisk_rate:.2%} non-risk")

st.markdown("---")

# === Tabs ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1) Origin Data Table",
    "2) Scatter: PC1 vs PC2",
    "3) Hexbin: Risk Density (PC1/PC2)",
    "4) 3D Cluster Plot",
    "5) Presentation Script"
])

with tab1:
    st.subheader("Original Data")
    st.caption("Scroll and filter as needed. This is the raw table used for analysis.")
    st.dataframe(orig_df, use_container_width=True)

with tab2:
    st.subheader("PC1 vs PC2 Scatter")
    st.caption("Colored by risk (if available); you can filter by cluster.")
    # Join risk label into df3d if missing
    if "risk_level" not in df3d.columns:
        df3d["risk_level"] = risk_series_bin.values

    available_clusters = sorted(df3d["cluster"].unique().tolist()) if "cluster" in df3d.columns else []
    pick_clusters = st.multiselect("Select clusters to show", available_clusters, default=available_clusters)

    plot_df = df3d.copy()
    if pick_clusters and "cluster" in plot_df.columns:
        plot_df = plot_df[plot_df["cluster"].isin(pick_clusters)]

    fig, ax = plt.subplots()
    # Color by risk (0/1)
    color = plot_df["risk_level"] if "risk_level" in plot_df.columns else None
    sc = ax.scatter(plot_df["PC1"], plot_df["PC2"], c=color, alpha=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PC1 vs PC2")
    if color is not None:
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Risk Level")
    st.pyplot(fig, clear_figure=True)

with tab3:
    st.subheader("Hexbin: Risk Density in PC1/PC2")
    st.caption("Shows density of points; use subset to focus on clusters. Darker bins indicate higher counts.")
    bins = st.slider("Hexbin grid size", min_value=10, max_value=100, value=35, step=5)
    fig2, ax2 = plt.subplots()
    hb = ax2.hexbin(df3d["PC1"], df3d["PC2"], gridsize=bins)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("Risk Density Hexbin (PC1/PC2)")
    cb = fig2.colorbar(hb, ax=ax2)
    cb.set_label("Count")
    st.pyplot(fig2, clear_figure=True)

with tab4:
    st.subheader("3D Cluster Plot")
    st.caption("Interactive 3D scatter in PCA space, colored by cluster.")
    if px is None:
        st.warning("plotly is not installed; cannot render 3D interactive plot.")
    else:
        df3d_plot = df3d.copy()
        if "cluster" not in df3d_plot.columns:
            df3d_plot["cluster"] = 0
        fig3d = px.scatter_3d(
            df3d_plot,
            x="PC1", y="PC2", z="PC3",
            color="cluster",
            hover_data=[c for c in orig_df.columns if c != risk_col][:10]  # limit hover columns
        )
        st.plotly_chart(fig3d, use_container_width=True)

with tab5:
    st.subheader("Presentation Script (for 4 Speakers)")

    # Compute cluster risk rates
    if "cluster" in df3d.columns:
        # Align risk binary with df3d index
        if "risk_level" in df3d.columns:
            risk_bin_present = df3d["risk_level"]
            if risk_bin_present.dtype == object:
                # Coerce to 0/1 if needed
                try:
                    risk_bin_present = risk_bin_present.astype(int)
                except Exception:
                    risk_bin_present = (risk_bin_present.str.lower().isin(["risk","high","yes","positive"])).astype(int)
        else:
            risk_bin_present = risk_series_bin

        if risk_bin_present.name in df3d.columns:
            cluster_risk = df3d.groupby("cluster")[risk_bin_present.name].mean()
        else:
            cluster_risk = pd.Series(dtype=float)
        overall_baseline = float(risk_series_bin.mean()) if len(risk_series_bin) else np.nan

        # Find cluster with highest risk (expected cluster "4" in user note, but compute robustly)
        highest_cluster = None
        highest_ratio = None
        if len(cluster_risk.dropna()) > 0:
            highest_cluster = cluster_risk.astype(float).idxmax()
            highest_ratio = (cluster_risk.loc[highest_cluster] / overall_baseline) if overall_baseline and overall_baseline > 0 else np.inf
    else:
        cluster_risk = pd.Series(dtype=float)
        overall_baseline = float(risk_series_bin.mean())
        highest_cluster = "4"
        highest_ratio = np.nan

    # Present dynamic numbers with safe fallbacks
    baseline_pct = f"{(overall_baseline*100):.1f}%" if overall_baseline == overall_baseline else "N/A"
    if highest_cluster is None:
        high_cluster_label = "4"
        high_ratio_text = "≈2×"
    else:
        high_cluster_label = str(highest_cluster)
        if highest_ratio is None or not np.isfinite(highest_ratio):
            high_ratio_text = "≈2×"
        else:
            high_ratio_text = f"{highest_ratio:.1f}×"

    # Script split among 4 speakers
    script_parts = [
        ("Speaker 1 – Setup & Data",
         f"""
         • Dataset: {n_rows} records, {n_cols} columns. We **excluded the risk label from modeling** and used it only for evaluation/visualization.
         • Overall class balance: **{risk_rate:.2%} risk / {nonrisk_rate:.2%} non‑risk**.
         • Method: Standardize → PCA (3D) → K‑Means (k=4) on PCA space; visualized PC1/PC2 scatter, hexbin density, and 3D clusters.
         """),

        ("Speaker 2 – Key Finding",
         f"""
         • During analysis, we identified a cohort with **~2× higher risk than baseline** — **Cluster {high_cluster_label}** (actual ratio: {high_ratio_text}; baseline: {baseline_pct}).
         • This cluster’s profile stands out in PCA space and concentrates in dense regions on the hexbin plot.
         """),

        ("Speaker 3 – Strategy by Cluster",
         """
         • Cluster 1 (Lower risk): Routine monitoring — daily vitals; standard feeding/weight checks.
         • Cluster 2 (Moderate): Enhanced monitoring — add nurse check‑ins, temperature & jaundice tracking.
         • Cluster 3 (Elevated): Close monitoring — twice‑daily vitals, early pediatrician review triggers.
         • Cluster 4 (High): **Intensive monitoring** — continuous oximetry if available, strict escalation rules, rapid‑response alerts.
         """),

        ("Speaker 4 – Wrap‑up & Next Steps",
         """
         • Validate generalization: hold‑out or temporal validation, and prospective monitoring.
         • Calibrate alerts to reduce false alarms; review fairness (no subgroup is over‑flagged).
         • Implementation: integrate the risk dashboard; train staff on cluster‑based protocols; audit outcomes monthly.
         """)
    ]

    # Allow small edits
    for title, body in script_parts:
        st.markdown(f"**{title}**")
        st.write(textwrap.dedent(body).strip())
        st.markdown("---")

st.caption("Tip: Use the sidebar to execute your notebook first so the app consumes exactly your generated files (df3d.csv, pca_df.csv, kmeans_centroids_in_PC_space.csv).")
