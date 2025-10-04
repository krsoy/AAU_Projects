import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
# Mahesh start

f_df= pd.read_csv('newborn_health_monitoring_with_risk.csv')

# drop apgar_score
f_df.drop(columns=['apgar_score'],inplace=True)
# get rid of risk level influence

# drop rows with null values
f_df.dropna(inplace=True)

# reset index
f_df.reset_index(inplace=True,drop=True)
f_df['risk_level'] = f_df['risk_level'].map({'At Risk':1,'Healthy':0})
risk_level = f_df['risk_level']
f_df.drop(columns=['risk_level'],inplace=True)

data_for_analysis = f_df.select_dtypes(include=['number'])
data_for_analysis.dropna(inplace=True)
data_for_analysis.reset_index(inplace=True,drop=True)

# we need to scale the data help the algorithm not to weight some "big" number data.
from sklearn.preprocessing import MinMaxScaler
scaler_minmax = MinMaxScaler()
data_minmax_scaled = scaler_minmax.fit_transform(data_for_analysis)
data_minmax_df = pd.DataFrame(data_minmax_scaled, columns=data_for_analysis.columns)

print("\nMinMax Scaling: Range = [0,1]")
print(f"birth_weight_kg - Min: {data_minmax_df['birth_weight_kg'].min():.3f}, Max: {data_minmax_df['birth_weight_kg'].max():.3f}")

# init PCA and feed data into it
from sklearn.decomposition import PCA
pca = PCA()
pca_results = pca.fit_transform(data_minmax_df)
pca_with_cluster = None
# Examine explained variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print("Explained Variance by Component:")
for i in range(min(10, len(explained_variance_ratio))):
    print(f"PC{i+1}: {explained_variance_ratio[i]:.3f} ({explained_variance_ratio[i]*100:.1f}%)")

print(f"\nFirst 3 components explain {cumulative_variance[2]*100:.1f}% of total variance")
print(f"First 5 components explain {cumulative_variance[4]*100:.1f}% of total variance")

# Mahesh end

st.set_page_config(page_title="PCA Variance Explorer", layout="wide")

# --- Expect these to be defined upstream in your notebook/session ---
# explained_variance_ratio: np.ndarray shape (n_components,)
# cumulative_variance: np.ndarray shape (n_components,)  -> optional; will be computed if missing.

# If cumulative_variance isn't defined, compute it safely:
if "cumulative_variance" not in globals() or "explained_variance_ratio" not in globals():
    st.error("Please ensure `explained_variance_ratio` is defined in the session before running this app.")
    st.stop()

if "cumulative_variance" not in globals() or cumulative_variance is None:
    cumulative_variance = np.cumsum(explained_variance_ratio)

pca_df = pd.DataFrame(pca_results[:, :3], columns=['PC1', 'PC2', 'PC3'])

# Attach key clinical fields for profiling (keep only those that exist)
attach_cols = [
    'age_days', 'jaundice_level_mg_dl', 'feeding_frequency_per_day',
    'stool_count', 'urine_output_count', 'weight_kg', 'length_cm',
    'head_circumference_cm', 'oxygen_saturation', 'temperature_c',
    'heart_rate_bpm', 'respiratory_rate_bpm'
]
attach_cols = [c for c in attach_cols if c in data_for_analysis.columns]
pca_df = pd.concat([pca_df, data_for_analysis[attach_cols]], axis=1)
from scipy.stats import pointbiserialr

st.set_page_config(page_title="Newborn EDA", layout="wide")

# --- Tabs ---
tabs = st.tabs(["Exploratory Data Analysis", "Variance Analysis", "risk_level vs. PCA spaces", "Cluster Visualization", "Summary"])

# =============== TAB 0: Exploratory Data Analysis ===============
with tabs[0]:
    st.subheader("Exploratory Data Analysis")

    # --- Load data ---
    df = f_df

    # --- Basic dataset info ---
    st.markdown("### Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows", len(df))
    with c2:
        st.metric("Columns", df.shape[1])
    with c3:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.metric("Numeric columns", len(num_cols))
    with c4:
        cat_cols = [c for c in df.columns if c not in num_cols]
        st.metric("Non-numeric columns", len(cat_cols))

    st.markdown("#### Peek at data")
    st.dataframe(df.head(10), use_container_width=True)

    # --- Baseline risk rate (overall and by health) ---
    st.markdown("### Baseline risk rate")
    overall_rate = risk_level.mean()
    c1, c2 = st.columns(2)
    c1.metric("Overall risk rate", f"{overall_rate:.2%}")

    st.divider()

    # --- Groupby health and risk, bar plot of average differences across columns ---
    st.markdown("### Groupby health & risk: feature differences")

    column_selector = st.selectbox(options=data_for_analysis.columns, label="Select column to analyze")
    value_slider_range = st.slider(min_value=float(data_for_analysis[column_selector].min()), max_value=float(data_for_analysis[column_selector].max()), value=(float(data_for_analysis[column_selector].min()), float(data_for_analysis[column_selector].max())), step=0.1, label="Select value range to filter")
    filtered_df = data_for_analysis[(data_for_analysis[column_selector] >= value_slider_range[0]) & (data_for_analysis[column_selector] <= value_slider_range[1])]
    # using filtered df to calculate current average risk level
    current_risk_level = risk_level[filtered_df.index]
    current_risk_rate = current_risk_level.mean()
    c2.metric(f"Risk rate for {column_selector} in [{value_slider_range[0]}, {value_slider_range[1]}]", f"{current_risk_rate:.2%}")
    st.markdown(f"**Note:** Current risk rate is based on filtering `{column_selector}` in the range [{value_slider_range[0]}, {value_slider_range[1]}].")

    # plot a streamlit bar chart that shows the average of each numeric column grouped by risk_level, but risk_level should be string healthy or at risk
    avg_by_risk = pd.concat([filtered_df, current_risk_level], axis=1).groupby(current_risk_level).mean().T
    avg_by_risk.columns = ['Healthy', 'At Risk']
    st.bar_chart(avg_by_risk, use_container_width=True)


with tabs[1]:
    st.subheader("Explained & Cumulative Variance")

    # Slider controls the target cumulative variance percentage (horizontal line)
    pct = st.slider(
        "Target cumulative variance (%)",
        min_value=50,
        max_value=99,
        value=80,
        step=1,
        help="Move the slider to set the target cumulative variance line."
    )
    thr = pct / 100.0

    # How many components are needed to reach the threshold?
    # np.searchsorted finds the first index where cumulative_variance >= thr
    # +1 to convert zero-based index to component count
    n_components_needed = int(np.searchsorted(cumulative_variance, thr) + 1)

    # Limit to first 15 for the view (like your original)
    k = int(min(15, len(explained_variance_ratio)))
    x_axis = range(1, k + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Scree plot
    ax1.plot(
        x_axis,
        explained_variance_ratio[:k],
        'bo-',
        linewidth=2,
        markersize=8
    )
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Scree Plot: Variance Explained by Each Component')
    ax1.grid(True, alpha=0.3)

    # Cumulative variance with dynamic threshold line
    ax2.plot(
        x_axis,
        cumulative_variance[:k],
        'ro-',
        linewidth=2,
        markersize=8
    )
    ax2.axhline(y=thr, color='gray', linestyle='--', alpha=0.8, label=f'{pct}% Variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig)

    st.markdown(
        f"**Components needed to reach {pct}% variance:** `{n_components_needed}` "
        f"(cumulative variance at that component: "
        f"{cumulative_variance[n_components_needed-1]:.3f})"
    )

    st.subheader("Principal Component Loadings & Projection")

    # Safety checks
    missing = []
    for name in ["pca", "explained_variance_ratio", "pca_results", "data_for_analysis"]:
        if name not in globals():
            missing.append(name)
    if missing:
        st.error(f"Missing variables in session: {', '.join(missing)}. Please run the PCA step first.")
        st.stop()

    # Build components_df (first up to 5 PCs)
    n_avail_pcs = min(5, pca.components_.shape[0])
    components_df = pd.DataFrame(
        pca.components_[:n_avail_pcs].T,
        columns=[f'PC{i + 1}' for i in range(n_avail_pcs)],
        index=data_for_analysis.columns
    )

    st.markdown("**Principal Component Loadings (each variable’s contribution to each PC):**")
    st.dataframe(components_df.round(3), use_container_width=True)

    # Controls
    st.markdown("**Display options**")
    left, mid, right = st.columns([2, 1, 1])
    with left:
        top_k = st.slider(
            "Top variables per PC by |loading|",
            min_value=5,
            max_value=min(25, components_df.shape[0]),
            value=min(12, components_df.shape[0]),
            help="Controls how many strongest-loading variables to show per PC."
        )
    with mid:
        show_labels = st.checkbox("Label points in PC1–PC2 scatter", value=False)
    with right:
        max_labels = st.number_input(
            "Max labels (if enabled)",
            min_value=5, max_value=300, value=50, step=5,
            help="Upper bound on how many points get text labels."
        )

    # Prepare figure (higher DPI for sharpness; constrained layout to reduce overlap)
    fig, axes = plt.subplots(
        2, 2,
        figsize=(10, 9),
        constrained_layout=True
    )


    # Helper to draw one PC’s horizontal bar chart
    def draw_pc_bar(ax, pc_idx):
        pc_name = f"PC{pc_idx + 1}"
        if pc_name not in components_df.columns:
            ax.axis("off")
            return
        series = components_df[pc_name]
        # pick top-k by |loading|
        series = series.reindex(series.abs().sort_values(ascending=False).head(top_k).index)
        series = series.sort_values()  # small->large for tidy barh
        bars = ax.barh(range(len(series)), series.values, linewidth=0.6)
        ax.set_yticks(range(len(series)))
        ax.set_yticklabels(series.index, fontsize=8)
        ax.invert_yaxis()  # largest at top
        var_pct = explained_variance_ratio[pc_idx] * 100 if pc_idx < len(explained_variance_ratio) else np.nan
        ax.set_title(f'{pc_name} Loadings (Explains {var_pct:.1f}% of variance)')
        ax.axvline(x=0, linestyle='-', alpha=0.35, linewidth=0.8)
        ax.margins(y=0.02)
        ax.grid(True, alpha=0.25, linewidth=0.5)

        # Subtle edges for crispness
        for b in bars:
            b.set_edgecolor('black')
            b.set_linewidth(0.4)
            b.set_alpha(0.9)


    # PC1, PC2, PC3 loadings (only if available)
    draw_pc_bar(axes[0, 0], 0)
    draw_pc_bar(axes[0, 1], 1)
    draw_pc_bar(axes[1, 0], 2)

    # PC1 vs PC2 scatter
    ax = axes[1, 1]
    if pca_results.shape[1] >= 2:
        # leaner markers for a less "vivid" look
        ax.scatter(
            pca_results[:, 0],
            pca_results[:, 1],
            s=14, alpha=0.7, linewidths=0.3
        )
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Samples in PC1–PC2 Space')
        ax.grid(True, alpha=0.25, linewidth=0.5)

        # optional labels (limit collisions by labeling the most "extreme" points)
        if show_labels:
            try:
                # priority by radial distance from origin in PC1-PC2 plane
                xy = pca_results[:, :2]
                dist = np.sqrt((xy ** 2).sum(axis=1))
                order = np.argsort(-dist)[: int(max_labels)]
                labels = getattr(data_for_analysis, "index", pd.RangeIndex(len(pca_results)))
                # alternate slight offsets to reduce collisions
                offsets = [(3, 2), (-3, -2), (4, -2), (-4, 2)]
                for i, idx in enumerate(order):
                    dx, dy = offsets[i % len(offsets)]
                    ax.annotate(
                        str(labels[idx]),
                        (xy[idx, 0], xy[idx, 1]),
                        xytext=(dx, dy),
                        textcoords="offset points",
                        fontsize=7,
                        alpha=0.9,
                        ha="left", va="bottom"
                    )
            except Exception:
                pass
    else:
        ax.axis("off")
        st.warning("pca_results has fewer than 2 components; cannot draw PC1–PC2 scatter.")

    # Apply tight layout as a final pass
    try:
        fig.tight_layout()
    except Exception:
        pass

    st.pyplot(fig, clear_figure=True, use_container_width=True)


with tabs[2]:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    st.subheader("Validation: risk_level vs. PCA spaces")

    # --- Safety checks ---
    missing = []
    for name in ["pca_df", "risk_level"]:
        if name not in globals():
            missing.append(name)
    if missing:
        st.error(f"Missing variables in session: {', '.join(missing)}. "
                 "Please ensure PCA results (`pca_df`) and `risk_level` are defined.")
        st.stop()

    # Align a copy and attach risk_level JUST FOR VALIDATION
    _pca_df = pca_df.copy()
    try:
        _pca_df["risk_level"] = pd.Series(risk_level, index=_pca_df.index).values
    except Exception as e:
        st.error(f"Failed to align `risk_level` with `pca_df`: {e}")
        st.stop()

    # Basic sanity + info
    st.markdown("**risk_level value counts** (for validation only):")
    st.write(_pca_df["risk_level"].value_counts(dropna=False))

    classes = sorted(_pca_df["risk_level"].dropna().unique())
    is_binary = (len(classes) == 2)
    baseline_prevalence = None
    if is_binary:
        try:
            baseline_prevalence = float((_pca_df["risk_level"] == 1).mean())
        except Exception:
            pass

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Samples", len(_pca_df))
    with c2:
        st.metric("Distinct risk classes", len(classes))
    with c3:
        st.metric("Baseline risk prevalence", f"{baseline_prevalence:.2%}" if baseline_prevalence is not None else "—")

    st.info("`risk_level` is reused **only** for validation/visualization. Do **not** feed it into model training.")

    # --- Controls ---
    st.markdown("**Display options**")
    cc1, cc2, cc3, cc4 = st.columns([1,1,1,1])
    with cc1:
        pt_size = st.slider("Scatter point size", 6, 24, 12, step=2)
    with cc2:
        pt_alpha = st.slider("Scatter alpha", 0.2, 1.0, 0.7, step=0.1)
    with cc3:
        grid_size = st.slider("Hexbin gridsize", 10, 80, 40, step=2)
    with cc4:
        vmax_override = st.checkbox("Clamp hexbin to [0,1]", value=True)

    # -------------------- Side-by-side: Scatter & Hexbin --------------------
    st.markdown("### PC1 × PC2 — Scatter vs. Hexbin (side-by-side)")
    col_scatter, col_hex = st.columns(2, gap="medium")

    # Data
    xvals = _pca_df["PC1"].values
    yvals = _pca_df["PC2"].values
    cvals = _pca_df["risk_level"].values

    # Scatter (left)
    with col_scatter:
        fig1, ax1 = plt.subplots(figsize=(6, 5), dpi=150, constrained_layout=True)
        if is_binary:
            cmap = ListedColormap(["#7aa2ff", "#ff7a7a"])
            sc = ax1.scatter(
                xvals, yvals,
                c=(cvals > 0).astype(int),
                s=pt_size, alpha=pt_alpha, cmap=cmap, linewidths=0.3
            )
            cbar = fig1.colorbar(sc, ax=ax1, ticks=[0, 1])
            cbar.ax.set_yticklabels(["Non-risk (0)", "Risk (1)"])
        else:
            sc = ax1.scatter(xvals, yvals, c=cvals, s=pt_size, alpha=pt_alpha, linewidths=0.3)
            cbar = fig1.colorbar(sc, ax=ax1)
            cbar.set_label("risk_level")

        ax1.axvline(0, lw=0.8, ls='--', alpha=0.6)
        ax1.axhline(0, lw=0.8, ls='--', alpha=0.6)
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.set_title("PC1 × PC2 Colored by risk_level")
        ax1.grid(True, alpha=0.25, linewidth=0.5)
        st.pyplot(fig1, use_container_width=True, clear_figure=True)

    # Hexbin (right)
    with col_hex:
        df_hex = _pca_df.dropna(subset=["PC1", "PC2", "risk_level"]).copy()
        x = df_hex["PC1"].values
        y = df_hex["PC2"].values
        z = df_hex["risk_level"].values

        fig2, ax2 = plt.subplots(figsize=(6, 5), dpi=150, constrained_layout=True)
        hb = ax2.hexbin(
            x, y,
            C=z,
            reduce_C_function=np.mean,
            gridsize=grid_size,
            mincnt=1,
            cmap="viridis",
            vmin=0 if vmax_override else None,
            vmax=1 if vmax_override else None,
        )
        cbar2 = fig2.colorbar(hb, ax=ax2)
        cbar2.set_label("High-risk prevalence")
        ax2.axvline(0, lw=0.8, ls='--', color='k', alpha=0.4)
        ax2.axhline(0, lw=0.8, ls='--', color='k', alpha=0.4)
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_title("Risk prevalence in PC1 × PC2 (hexbin mean of 0/1)")
        ax2.grid(True, alpha=0.2, linewidth=0.4)
        st.pyplot(fig2, use_container_width=True, clear_figure=True)

    # -------------------- Boxplots of PCs by risk_level --------------------
    st.markdown("### Distribution of PCs by `risk_level` (boxplots)")
    pc_candidates = [c for c in _pca_df.columns if c.startswith("PC")]
    default_pcs = [c for c in ["PC1", "PC2", "PC3"] if c in pc_candidates]
    sel_pcs = st.multiselect(
        "Select PCs to plot",
        options=pc_candidates,
        default=default_pcs if default_pcs else pc_candidates[:3]
    )

    if len(sel_pcs) == 0:
        st.info("Select at least one PC to plot.")
    else:
        fig3, axes3 = plt.subplots(
            1, len(sel_pcs),
            figsize=(4.5 * len(sel_pcs), 4.2),
            dpi=150,
            constrained_layout=True
        )
        axes_list = [axes3] if len(sel_pcs) == 1 else list(axes3)
        for ax, pc in zip(axes_list, sel_pcs):
            try:
                _pca_df.boxplot(column=pc, by="risk_level", ax=ax)
                ax.set_title(f"{pc} by risk_level")
                ax.set_xlabel("risk_level")
                ax.set_ylabel(pc)
                ax.grid(True, alpha=0.25, linewidth=0.5)
            except Exception as e:
                ax.axis("off")
                ax.set_title(f"Failed to plot {pc}: {e}")

        try:
            fig3.suptitle("")
            fig3.tight_layout()
        except Exception:
            pass
        st.pyplot(fig3, use_container_width=True, clear_figure=True)

# =============== TAB 3: PCA 3D Visualization (side-by-side + comparison coloring) ===============
with tabs[3]:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from matplotlib.colors import ListedColormap
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score, silhouette_samples

    st.subheader("PCA 2D + 3D Visualization (risk-free clustering)")

    # -------- Safety checks --------

    # --- Optional columns detection for comparison modes ---
    def detect_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    health_col = detect_col(pca_df, ["health", "health_status", "status", "healthgroup", "health_group"])
    # risk_level may live outside pca_df; we’ll try both
    risk_series = None
    try:
        risk_series = pd.Series(risk_level, index=pca_df.index)
    except Exception:
        # maybe it's already inside pca_df
        if "risk_level" in pca_df.columns:
            risk_series = pca_df["risk_level"]

    # -------- Clustering controls --------
    st.markdown("**Clustering setup**")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        cum_target = st.slider("Target cumulative variance for PCs", 0.60, 0.99, 0.85, 0.01,
                               help="Use PCs until cumulative explained variance ≥ this value (capped at 6 PCs).")
    with cc2:
        k_min, k_max = st.select_slider("K range (min,max)", options=list(range(2, 11)), value=(2, 8))
    with cc3:
        st.write("")  # spacer
        st.caption("Both KMeans and Ward are evaluated; best silhouette is selected.")

    # --- Choose PCs (cap at 6) ---
    try:
        evr = np.asarray(explained_variance_ratio)
        cum = evr.cumsum()
        n_pc = min(int(np.searchsorted(cum, cum_target) + 1),
                   min(6, pca_df.filter(like='PC').shape[1]))
    except Exception:
        n_pc = min(3, pca_df.filter(like='PC').shape[1])

    pc_cols = [f'PC{i}' for i in range(1, n_pc + 1)]
    st.caption(f"Using **{n_pc} PCs**: {pc_cols}")

    # --- Scale and select best clustering by silhouette ---
    X = StandardScaler().fit_transform(pca_df[pc_cols].values)

    rows = []
    best = {'method': None, 'K': None, 'sil': -1, 'labels': None, 'model': None}
    for K in range(k_min, k_max + 1):
        # KMeans
        km = KMeans(n_clusters=K, n_init=50, random_state=42)
        lab_km = km.fit_predict(X)
        sil_km = silhouette_score(X, lab_km)
        rows.append(['kmeans', K, sil_km])
        if sil_km > best['sil']:
            best.update({'method': 'kmeans', 'K': K, 'sil': sil_km, 'labels': lab_km, 'model': km})

        # Ward
        ag = AgglomerativeClustering(n_clusters=K, linkage='ward')
        lab_wd = ag.fit_predict(X)
        sil_wd = silhouette_score(X, lab_wd)
        rows.append(['ward', K, sil_wd])
        if sil_wd > best['sil']:
            best.update({'method': 'ward', 'K': K, 'sil': sil_wd, 'labels': lab_wd, 'model': ag})

    sel_df = pd.DataFrame(rows, columns=['method', 'K', 'silhouette']).pivot(
        index='K', columns='method', values='silhouette'
    )
    st.markdown("**Silhouette by K (higher is better)**")
    st.dataframe(sel_df.round(3), use_container_width=True)

    st.success(
        f"Selected → **{best['method'].upper()}** with **K={best['K']}** "
        f"(silhouette = **{best['sil']:.3f}**) on **{n_pc} PCs**"
    )

    # Work on a copy with cluster labels
    df_vis = pca_df.copy()
    df_vis['cluster'] = best['labels']
    pca_with_cluster = df_vis.copy()
    # make df_vis global useage
    globals()['_df_vis'] = df_vis


    # --- Quality and profiling (expanders) ---
    sil_vals = silhouette_samples(X, df_vis['cluster'].values)
    sil_summary = (
        pd.DataFrame({'cluster': df_vis['cluster'], 'sil': sil_vals})
        .groupby('cluster')['sil'].agg(['count', 'mean', 'median', 'min', 'max'])
        .round(3)
    )
    with st.expander("Silhouette summary by cluster"):
        st.dataframe(sil_summary, use_container_width=True)

    clinical_cols = [
        'age_days','jaundice_level_mg_dl','feeding_frequency_per_day','stool_count','urine_output_count',
        'weight_kg','length_cm','head_circumference_cm','temperature_c','heart_rate_bpm','respiratory_rate_bpm'
    ]
    clinical_cols = [c for c in clinical_cols if c in df_vis.columns]
    if clinical_cols:
        prof = df_vis.groupby('cluster')[clinical_cols + pc_cols].median().round(2)
        with st.expander("Cluster median profiles (clinical + PCs)"):
            st.dataframe(prof, use_container_width=True)

    # -------- Coloring mode controls --------
    st.markdown("**Color by**")
    col_left, col_right = st.columns([1, 2])
    with col_left:
        color_mode = st.radio(
            "Select",
            ["Cluster", "Risk vs Non-risk"],
            index=0,
            help="Comparison modes use red/black only."
        )
    with col_right:
        point_size = st.slider("Point size", 3, 12, 5)
        point_alpha = st.slider("Point alpha", 0.2, 1.0, 0.85, 0.05)

    # Build color arrays for comparison modes (red/black only)
    # red = 'positive' class; black = others
    red_black = np.array(["black", "red"])

    # Helper: get risk binary (0/1) if available
    risk_bin = None
    if color_mode == "Risk vs Non-risk":
        if risk_series is None:
            st.warning("`risk_level` not found; falling back to cluster colors.")
            color_mode = "Cluster"
        else:
            # Try to coerce to 0/1
            r = risk_series.copy()
            if not np.issubdtype(r.dtype, np.number):
                mapping = {
                    "yes": 1, "y": 1, "true": 1, "t": 1, "risk": 1, "at_risk": 1, "high": 1, "1": 1,
                    "no": 0, "n": 0, "false": 0, "f": 0, "non-risk": 0, "low": 0, "0": 0
                }
                r = r.astype(str).str.lower().map(mapping).fillna(0).astype(int)
            else:
                r = (r > 0).astype(int)
            risk_bin = r.reindex(df_vis.index).astype(int)



    # -------- Side-by-side plots --------
    p2d, p3d = st.columns(2, gap="large")

    # === 2D matplotlib (PC1 vs PC2) ===
    with p2d:
        st.markdown("**PC1 vs PC2 (2D)**")
        fig2d, ax2d = plt.subplots(figsize=(5.6, 5.0), dpi=160, constrained_layout=True)

        if color_mode == "Cluster":
            sc = ax2d.scatter(df_vis['PC1'], df_vis['PC2'],
                              c=df_vis['cluster'], s=point_size*6, alpha=point_alpha, cmap='tab10', linewidths=0.3)
            cbar = fig2d.colorbar(sc, ax=ax2d)
            cbar.set_label("cluster")
        elif color_mode == "Risk vs Non-risk":
            colors = red_black[risk_bin.values]
            ax2d.scatter(df_vis['PC1'], df_vis['PC2'],
                         c=colors, s=point_size*6, alpha=point_alpha, linewidths=0.3)
            # legend proxies
            ax2d.scatter([], [], c="red", label="Risk (1)")
            ax2d.scatter([], [], c="black", label="Non-risk (0)")
            ax2d.legend(loc="best", frameon=False)
        else:  # Health (choose group)
            pass

        ax2d.axvline(0, ls='--', lw=0.8, color='grey'); ax2d.axhline(0, ls='--', lw=0.8, color='grey')
        ax2d.set_xlabel('PC1'); ax2d.set_ylabel('PC2')
        ax2d.grid(True, alpha=0.25, linewidth=0.5)
        st.pyplot(fig2d, use_container_width=True, clear_figure=True)

    # === 3D Plotly (PC1, PC2, PC3) ===
    with p3d:
        st.markdown("**PC1–PC3 (3D)**")

        if all(col in df_vis.columns for col in ['PC1', 'PC2', 'PC3']):
            df3d = df_vis.copy()

            hover_cols = [
                'age_days','jaundice_level_mg_dl','feeding_frequency_per_day','stool_count','urine_output_count',
                'weight_kg','length_cm','head_circumference_cm','temperature_c','heart_rate_bpm','respiratory_rate_bpm'
            ]
            hover_cols = [c for c in hover_cols if c in df3d.columns]

            if color_mode == "Cluster":
                fig = px.scatter_3d(
                    df3d, x='PC1', y='PC2', z='PC3',
                    color=df3d['cluster'].astype(str),
                    hover_data=hover_cols,
                    opacity=point_alpha,
                    title=f'Clusters in PCA space — {best["method"].upper()} (K={best["K"]})'
                )
                fig.update_traces(marker=dict(size=point_size))
                # optional centroids for KMeans only
                try:
                    if best['method'] == 'kmeans' and len(pc_cols) >= 3:
                        scaler_pc = StandardScaler().fit(df_vis[pc_cols].values)
                        centers_pc = scaler_pc.inverse_transform(best['model'].cluster_centers_)
                        # pad if fewer than 3 PCs used
                        if centers_pc.shape[1] < 3:
                            pad = np.zeros((centers_pc.shape[0], 3 - centers_pc.shape[1]))
                            centers_pc = np.hstack([centers_pc, pad])
                        fig.add_trace(go.Scatter3d(
                            x=centers_pc[:, 0], y=centers_pc[:, 1], z=centers_pc[:, 2],
                            mode='markers+text',
                            marker=dict(size=max(point_size+3, 6), color='black'),
                            text=[f'C{i}' for i in range(centers_pc.shape[0])],
                            textposition='top center',
                            name='Centroids'
                        ))
                except Exception:
                    pass

            elif color_mode == "Risk vs Non-risk":
                if risk_bin is None:
                    st.warning("`risk_level` not found; cannot show risk comparison.")
                    st.stop()
                color_list = red_black[risk_bin.values]
                fig = go.Figure(data=[go.Scatter3d(
                    x=df3d['PC1'], y=df3d['PC2'], z=df3d['PC3'],
                    mode='markers',
                    marker=dict(size=point_size, opacity=point_alpha, color=color_list),
                    text=None,
                    hovertemplate="<b>PC1</b>: %{x:.3f}<br><b>PC2</b>: %{y:.3f}<br><b>PC3</b>: %{z:.3f}<extra></extra>"
                )])
                fig.update_layout(
                    title="Risk vs Non-risk (red/black)",
                    showlegend=False
                )

            else:  # Health (choose group)
               pass

            fig.update_layout(
                legend_title_text='cluster' if color_mode == "Cluster" else None,
                scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
                margin=dict(l=0, r=0, t=40, b=0),
                height=520
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need PC1, PC2, and PC3 to render the 3D scatter.")


# =============== TAB 4: Summary & Decision Support ===============
with tabs[4]:
    import numpy as np
    import pandas as pd
    import streamlit as st

    st.subheader("Summary of Findings & Decision Support")

    # ---- Safety / inputs ----
    if 'pca_df' not in globals() or 'cluster' not in pca_with_cluster.columns:
        st.error("Missing `pca_df` with `cluster` labels. Please run clustering first (Tab 3).")
        st.stop()

    # risk_level may be external or already in pca_df
    risk_series = None
    try:
        risk_series = pd.Series(risk_level, index=pca_with_cluster.index)
    except Exception:
        if "risk_level" in pca_with_cluster.columns:
            risk_series = pca_with_cluster["risk_level"]

    if risk_series is None:
        st.warning("`risk_level` not found. Summary will exclude risk rates.")
        # Minimal info
        st.dataframe(
            pca_with_cluster['cluster'].value_counts().rename_axis('cluster')
                  .to_frame('count').sort_index(),
            use_container_width=True
        )
        st.stop()

    # ---- Normalize risk to 0/1 (just in case) ----
    r = risk_series.copy()
    if not np.issubdtype(r.dtype, np.number):
        mapping = {
            "yes": 1, "y": 1, "true": 1, "t": 1, "risk": 1, "at_risk": 1, "high": 1, "1": 1,
            "no": 0, "n": 0, "false": 0, "f": 0, "non-risk": 0, "low": 0, "0": 0
        }
        r = r.astype(str).str.lower().map(mapping).fillna(0).astype(int)
    else:
        r = (r > 0).astype(int)
    r = r.reindex(pca_with_cluster.index)

    df = pca_with_cluster.copy()
    df["risk_bin"] = r

    # ---- KPIs ----
    total_n = len(df)
    baseline = df["risk_bin"].mean() if total_n else 0.0
    k_clusters = int(df["cluster"].nunique())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total samples", f"{total_n:,}")
    c2.metric("Baseline risk rate", f"{baseline:.2%}")
    c3.metric("Number of clusters", f"{k_clusters}")

    # ---- Risk by cluster ----
    grp = df.groupby("cluster").agg(
        count=("risk_bin", "size"),
        risk_rate=("risk_bin", "mean")
    ).sort_index()
    grp["expected_at_risk"] = (grp["risk_rate"] * grp["count"]).round(1)

    # Lift vs baseline (handle baseline 0)
    grp["lift_vs_baseline"] = np.where(
        baseline > 0,
        grp["risk_rate"] / baseline,
        np.nan
    )

    # Priority tiers (tweak thresholds if you like)
    def assign_priority(lift):
        if np.isnan(lift):
            return "Unknown"
        if lift >= 2.0:
            return "Critical"
        if lift >= 1.5:
            return "High"
        if lift >= 1.2:
            return "Medium"
        return "Routine"

    grp["priority"] = grp["lift_vs_baseline"].apply(assign_priority)

    # Rank table for quick decisions
    summary = grp.sort_values(["priority", "lift_vs_baseline", "risk_rate"], ascending=[True, False, False])
    st.markdown("### Cluster Risk Summary")
    st.dataframe(
        summary.style.format({
            "risk_rate": "{:.2%}",
            "lift_vs_baseline": "{:.2f}"
        }),
        use_container_width=True
    )

    # ---- Callouts / insights ----
    st.markdown("### Insights")
    hi = grp.sort_values("risk_rate", ascending=False).head(1)
    hi_cluster = hi.index[0]
    hi_rate = float(hi["risk_rate"].iloc[0])
    hi_lift = float(hi["lift_vs_baseline"].iloc[0])

    bullets = []
    bullets.append(f"- **Baseline risk rate** is **{baseline:.2%}** across **{total_n:,}** samples and **{k_clusters}** clusters.")
    bullets.append(f"- **Highest-risk cluster** is **Cluster {hi_cluster}** at **{hi_rate:.2%}** (lift **{hi_lift:.2f}×**).")
    # “twice the baseline” trigger
    twice = grp[grp["lift_vs_baseline"] >= 2.0].index.tolist()
    if twice:
        bullets.append(f"- ⚠️ Cluster(s) **{', '.join(map(str, twice))}** show **≥2×** the baseline risk — prioritize immediate monitoring.")
    # Which clusters are High / Critical
    focus = summary[summary["priority"].isin(["Critical", "High"])].index.tolist()
    if focus:
        bullets.append(f"- **Focus tiers**: {', '.join(map(lambda x: 'Cluster '+str(x), focus))} require elevated attention.")

    st.markdown("\n".join(bullets))

    # ---- Monitoring plan (data-driven) ----
    st.markdown("### Monitoring Priority Plan")
    st.caption("Suggested action levels derived from risk lift vs. baseline (configurable thresholds).")
    plan = summary.reset_index().rename(columns={"index": "cluster"})
    plan["monitoring_level"] = plan["priority"].map({
        "Critical": "Level 1: Intensive (real-time alerts, frequent checks)",
        "High": "Level 2: Enhanced (daily checks, targeted interventions)",
        "Medium": "Level 3: Standard+ (periodic checks, watchlist)",
        "Routine": "Level 4: Routine (baseline monitoring)",
        "Unknown": "Assess data quality (missing baseline)"
    })
    plan_display = plan[["cluster", "count", "risk_rate", "lift_vs_baseline", "priority", "monitoring_level"]]
    st.dataframe(
        plan_display.style.format({"risk_rate":"{:.2%}", "lift_vs_baseline":"{:.2f}"}),
        use_container_width=True
    )

    # ---- Quick decision numbers ----
    st.markdown("### Quick Numbers for Decisions")
    q1, q2, q3 = st.columns(3)
    top2 = grp.sort_values("risk_rate", ascending=False).head(2)
    total_expected = grp["expected_at_risk"].sum()
    q1.metric("Expected at-risk (sum)", f"{int(round(total_expected))}")
    q2.metric("Top-1 cluster expected at-risk", f"{int(round(float(top2['expected_at_risk'].iloc[0])))}")
    if len(top2) > 1:
        q3.metric("Top-2 cumulative expected at-risk", f"{int(round(top2['expected_at_risk'].sum()))}")
    else:
        q3.metric("Top-2 cumulative expected at-risk", "—")

    # ---- Optional download ----
    st.download_button(
        "Download cluster summary (CSV)",
        data=summary.reset_index().to_csv(index=False),
        file_name="cluster_risk_summary.csv",
        mime="text/csv"
    )

    # ---- Narrative (brief, editable) ----
    st.markdown("### One-paragraph Executive Summary")
    st.write(
        f"Using PCA features and silhouette-selected clustering, we observe a baseline risk of **{baseline:.2%}**. "
        f"**Cluster {hi_cluster}** exhibits the highest risk (**{hi_rate:.2%}**, **{hi_lift:.2f}×** baseline). "
        f"Clusters classified as **Critical/High** should receive prioritized monitoring and interventions, while "
        f"Medium/Routine tiers remain on standard follow-up. This stratification supports resource allocation "
        f"and early alerts for newborns most likely to be at risk."
    )

