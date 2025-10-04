# streamlit_app.py  (dark style)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="PCA 3D Clusters (Dark)", layout="wide")

# --- Optional: force a dark-ish Streamlit surface even if the user theme is light
st.markdown("""
<style>
/* Page background + text */
.main, .stApp {
  background-color: #0e1117 !important;
  color: #e3e3e3 !important;
}
section[data-testid="stSidebar"] {
  background-color: #0b0e14 !important;
}
div[data-testid="stMarkdownContainer"] h1, 
div[data-testid="stMarkdownContainer"] h2, 
div[data-testid="stMarkdownContainer"] h3 {
  color: #e3e3e3 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("PCA 3D Clusters — Interactive Viewer (Dark)")

# --- Load data (cached) ---
@st.cache_data
def load_data():
    df3d = pd.read_csv('df3d.csv')
    pca_df = pd.read_csv('pca_df.csv')
    centers = pd.read_csv('kmeans_centroids_in_PC_space.csv')  # already in PC units
    return df3d, pca_df, centers

df3d, pca_df, cluster_centers = load_data()
if 'cluster' in df3d.columns:
    df3d['cluster'] = df3d['cluster'].astype(str)

# --- Sidebar controls ---
st.sidebar.header("Controls")

available_pcs = [c for c in df3d.columns if c.startswith('PC')]
default_x = 'PC1' if 'PC1' in available_pcs else available_pcs[0]
default_y = 'PC2' if 'PC2' in available_pcs else available_pcs[min(1, len(available_pcs)-1)]
default_z = 'PC3' if 'PC3' in available_pcs else available_pcs[min(2, len(available_pcs)-1)]

x_pc = st.sidebar.selectbox("X axis PC", available_pcs, index=available_pcs.index(default_x))
y_pc = st.sidebar.selectbox("Y axis PC", available_pcs, index=available_pcs.index(default_y))
z_pc = st.sidebar.selectbox("Z axis PC", available_pcs, index=available_pcs.index(default_z))

if 'cluster' in df3d.columns:
    all_clusters = sorted(df3d['cluster'].unique().tolist(), key=lambda s: int(s) if s.isdigit() else s)
    show_clusters = st.sidebar.multiselect("Show clusters", all_clusters, default=all_clusters)
else:
    show_clusters = None

highlight_risk = st.sidebar.checkbox("Highlight risk_level in red", value=False)
size = st.sidebar.slider("Point size", 1, 8, 3)
opacity = st.sidebar.slider("Point opacity", 0.1, 1.0, 0.85)
add_jitter = st.sidebar.checkbox("Add tiny jitter (visual only)", value=False)
jitter_std = st.sidebar.slider("Jitter (std)", 0.0, 0.1, 0.03) if add_jitter else 0.0
show_centroids = st.sidebar.checkbox("Show K-means centroids", value=True)

# --- Prep data ---
plot_df = df3d.copy()
if show_clusters is not None:
    plot_df = plot_df[plot_df['cluster'].isin(show_clusters)]

if add_jitter:
    rng = np.random.default_rng(42)
    for col in [x_pc, y_pc, z_pc]:
        if col in plot_df.columns:
            plot_df[col] = plot_df[col].astype(float) + rng.normal(0.0, jitter_std, size=len(plot_df))

hover_cols = [
    'age_days','jaundice_level_mg_dl','feeding_frequency_per_day','stool_count','urine_output_count',
    'weight_kg','length_cm','head_circumference_cm','temperature_c','heart_rate_bpm','respiratory_rate_bpm'
]
hover_cols = [c for c in hover_cols if c in plot_df.columns]

# --- Build dark Plotly figure ---
template = "plotly_dark"
title = f"3D PCA Scatter — {x_pc}, {y_pc}, {z_pc}"

if highlight_risk and 'risk_level' in plot_df.columns:
    plot_df['_risk_str'] = plot_df['risk_level'].astype(int).astype(str)
    fig = px.scatter_3d(
        plot_df, x=x_pc, y=y_pc, z=z_pc,
        color='_risk_str',
        color_discrete_map={'0': '#b0b0b0', '1': '#ff3b3b'},
        hover_data=hover_cols + (['cluster'] if 'cluster' in plot_df.columns else []),
        opacity=opacity, template=template, title=title + " (risk overlay)"
    )
else:
    if 'cluster' in plot_df.columns:
        fig = px.scatter_3d(
            plot_df, x=x_pc, y=y_pc, z=z_pc,
            color='cluster', hover_data=hover_cols,
            opacity=opacity, template=template, title=title + " (by cluster)"
        )
    else:
        fig = px.scatter_3d(
            plot_df, x=x_pc, y=y_pc, z=z_pc,
            opacity=opacity, template=template, title=title
        )
fig.update_traces(marker=dict(size=size))

# Centroids overlay (in PC units)
if show_centroids:
    needed_cols = [x_pc, y_pc, z_pc]
    if all(c in cluster_centers.columns for c in needed_cols):
        centers_pc = cluster_centers[needed_cols].values
        fig.add_trace(go.Scatter3d(
            x=centers_pc[:, 0], y=centers_pc[:, 1], z=centers_pc[:, 2],
            mode='markers+text',
            marker=dict(size=10, color='white'),
            text=[f'C{i}' for i in range(centers_pc.shape[0])],
            textposition='top center',
            name='Centroids'
        ))
    else:
        st.warning(f"Centroids file missing one of the selected axes: {needed_cols}. "
                   f"Available: {', '.join(cluster_centers.columns)}")

# Solid dark backgrounds & font
fig.update_layout(
    width=None, height=720,
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(color="#e3e3e3"),
    scene=dict(
        xaxis_title=x_pc, yaxis_title=y_pc, zaxis_title=z_pc,
        xaxis=dict(backgroundcolor="#0e1117", gridcolor="#2a2f3a", zerolinecolor="#2a2f3a"),
        yaxis=dict(backgroundcolor="#0e1117", gridcolor="#2a2f3a", zerolinecolor="#2a2f3a"),
        zaxis=dict(backgroundcolor="#0e1117", gridcolor="#2a2f3a", zerolinecolor="#2a2f3a"),
    ),
    legend_title_text='risk_level' if highlight_risk else 'cluster'
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("Notes"):
    st.markdown(
        "- Dark theme via Plotly `template='plotly_dark'` + dark backgrounds.\n"
        "- Sidebar controls: PC axis mapping, cluster filter, risk overlay, jitter, centroids.\n"
        "- Risk overlay: **red = risk=1**, **gray = risk=0**.\n"
        "- Data expected: `df3d.csv`, `pca_df.csv`, `kmeans_centroids_in_PC_space.csv`."
    )
