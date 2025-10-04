import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
# Load data
ad_df = pd.read_csv('cardata/Ad_table (extra).csv')
# avegrage mpg is string 13.7 mpg, turn it into float
ad_df['Average_mpg'] = ad_df['Average_mpg'].str.replace(' mpg', '').astype(float)
ad_df['Top_speed'] = ad_df['Top_speed'].str.replace(' mph', '').astype(float)
# engin size is string 2.0L, turn it into float
ad_df['Engin_size'] = ad_df['Engin_size'].str.replace('L', '').astype(float)
# Annual_Tax to float
# Annual_Tax to float
ad_df['Annual_Tax'] = ad_df['Annual_Tax'].str.replace('*', '').replace('','0').astype(float)

# select only the float columns
data_for_analysis = ad_df.select_dtypes(include=[np.number])
data_for_analysis.dropna(inplace=True)
data_for_analysis.reset_index(inplace=True)
# we need to scale the data help the algorithm not to weight some "big" number data.
from sklearn.preprocessing import MinMaxScaler
scaler_minmax = MinMaxScaler()
data_minmax_scaled = scaler_minmax.fit_transform(data_for_analysis)
data_minmax_df = pd.DataFrame(data_minmax_scaled, columns=data_for_analysis.columns)

print("\nMinMax Scaling: Range = [0,1]")
print(f"Engine_power - Min: {data_minmax_df['Engine_power'].min():.3f}, Max: {data_minmax_df['Engine_power'].max():.3f}")
print(f"Price - Min: {data_minmax_df['Price'].min():.3f}, Max: {data_minmax_df['Price'].max():.3f}")

# init PCA and feed data into it
from sklearn.decomposition import PCA
pca = PCA()
pca_results = pca.fit_transform(data_minmax_df)

# Examine explained variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print("Explained Variance by Component:")
for i in range(min(10, len(explained_variance_ratio))):
    print(f"PC{i+1}: {explained_variance_ratio[i]:.3f} ({explained_variance_ratio[i]*100:.1f}%)")

print(f"\nFirst 3 components explain {cumulative_variance[2]*100:.1f}% of total variance")
print(f"First 5 components explain {cumulative_variance[4]*100:.1f}% of total variance")


def plot_1():
    # Visualize the explained variance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Scree plot
    ax1.plot(range(1, min(16, len(explained_variance_ratio) + 1)),
             explained_variance_ratio[:15], 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Scree Plot: Variance Explained by Each Component')
    ax1.grid(True, alpha=0.3)

    # Cumulative variance
    ax2.plot(range(1, min(16, len(cumulative_variance) + 1)),
             cumulative_variance[:15], 'ro-', linewidth=2, markersize=8)
    ax2.axhline(y=0.8, color='gray', linestyle='--', alpha=0.7, label='80% Variance')
    ax2.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% Variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

components_df = pd.DataFrame(
    pC[:5].T,  # First 5 components
    columns=[f'PC{i+1}' for i in range(5)],
    index=data_for_analysis.columns
)

print("Principal Component Loadings (How much each variable contributes):")
print(components_df.round(3))

def plot_2():
    # Visualize component loadings for interpretation
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # PC1 loadings
    pc1_loadings = components_df['PC1'].sort_values(key=abs, ascending=False)
    axes[0, 0].barh(range(len(pc1_loadings)), pc1_loadings.values)
    axes[0, 0].set_yticks(range(len(pc1_loadings)))
    axes[0, 0].set_yticklabels(pc1_loadings.index, fontsize=9)
    axes[0, 0].set_title(f'PC1 Loadings (Explains {explained_variance_ratio[0] * 100:.1f}% of variance)')
    axes[0, 0].axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # PC2 loadings
    pc2_loadings = components_df['PC2'].sort_values(key=abs, ascending=False)
    axes[0, 1].barh(range(len(pc2_loadings)), pc2_loadings.values, color='orange')
    axes[0, 1].set_yticks(range(len(pc2_loadings)))
    axes[0, 1].set_yticklabels(pc2_loadings.index, fontsize=9)
    axes[0, 1].set_title(f'PC2 Loadings (Explains {explained_variance_ratio[1] * 100:.1f}% of variance)')
    axes[0, 1].axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # PC3 loadings
    pc3_loadings = components_df['PC3'].sort_values(key=abs, ascending=False)
    axes[1, 0].barh(range(len(pc3_loadings)), pc3_loadings.values, color='green')
    axes[1, 0].set_yticks(range(len(pc3_loadings)))
    axes[1, 0].set_yticklabels(pc3_loadings.index, fontsize=9)
    axes[1, 0].set_title(f'PC3 Loadings (Explains {explained_variance_ratio[2] * 100:.1f}% of variance)')
    axes[1, 0].axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # PC1 vs PC2 scatter plot of cities
    axes[1, 1].scatter(pca_results[:, 0], pca_results[:, 1], alpha=0.6)
    axes[1, 1].set_xlabel('PC1')
    axes[1, 1].set_ylabel('PC2')
    axes[1, 1].set_title('Cities in PC1-PC2 Space')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Business interpretation of components
print("BUSINESS INTERPRETATION:")
print("="*50)

# Analyze PC1
pc1_positive = components_df['PC1'][components_df['PC1'] > 0.2].sort_values(ascending=False)
pc1_negative = components_df['PC1'][components_df['PC1'] < -0.2].sort_values()

print(f"PC1 (explains {explained_variance_ratio[0]*100:.1f}% of variance):")
print("High positive loadings (higher PC1 scores):", list(pc1_positive.index))
print("High negative loadings (lower PC1 scores):", list(pc1_negative.index))
print("Business meaning: Political Instability/Fragility vs Stability Dimension")
print("NOTE: fragile_states_index & press_freedom_index are INVERSE coded (higher = worse)")
print("High PC1 = Politically unstable, restricted countries")
print("Low PC1 = Stable democracies with high freedom")
print()

# Analyze PC2
pc2_positive = components_df['PC2'][components_df['PC2'] > 0.2].sort_values(ascending=False)
pc2_negative = components_df['PC2'][components_df['PC2'] < -0.2].sort_values()

print(f"PC2 (explains {explained_variance_ratio[1]*100:.1f}% of variance):")
print("High positive loadings (higher PC2 scores):", list(pc2_positive.index))
print("High negative loadings (lower PC2 scores):", list(pc2_negative.index))
print("meaning: epresents a contrast between old, high-engine-capacity, high-cost cars and newer, more practical family cars")
print("PC2 represents the vehicle “size–capacity–power” ")



# Create a dataframe with PCA results for easier analysis
pca_df = pd.DataFrame(
    pca_results[:, :3],  # First 3 components
    columns=['PC1', 'PC2', 'PC3']
)
pca_df['Door_num'] = data_for_analysis['Door_num']
pca_df['Reg_year'] = data_for_analysis['Reg_year']
pca_df['Annual_Tax'] = data_for_analysis['Annual_Tax']


# Find extreme cities for each component
print("CITIES AT EXTREMES OF EACH DIMENSION:")
print("="*50)

print("PC1 - Highest :budget, highly economical vehicles with moderate engine power.")
print(pca_df.nlargest(5, 'PC1')[['Door_num', 'Reg_year','Annual_Tax', 'PC1']].to_string(index=False))

print("\nPC1 - Lowest :extrem high/low engine power vehicles")
print(pca_df.nsmallest(5, 'PC1')[['Door_num', 'Reg_year','Annual_Tax', 'PC1']].to_string(index=False))

print("\nPC2 - Highest :I don't know about this yet")
print(pca_df.nlargest(5, 'PC2')[['Door_num', 'Reg_year','Annual_Tax', 'PC2']].to_string(index=False))

print("\nPC2 - Lowest: racing vehicles?")
print(pca_df.nsmallest(5, 'PC2')[['Door_num', 'Reg_year','Annual_Tax', 'PC2']].to_string(index=False))
