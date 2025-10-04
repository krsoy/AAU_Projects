import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

f_df= pd.read_csv('newborn_health_monitoring_with_risk.csv')
print(f_df.head())
print(f_df.info())
print(f_df.describe())
print(f_df.isnull().sum())
print(f_df.duplicated().sum())


# drop apgar_score
f_df.drop(columns=['apgar_score'],inplace=True)
# drop rows with null values
f_df.dropna(inplace=True)

# reset index
f_df.reset_index(inplace=True,drop=True)
f_df['risk_level'] = f_df['risk_level'].map({'At Risk':1,'Healthy':0})

data_for_analysis = f_df.select_dtypes(include=['number'])
data_for_analysis.dropna(inplace=True)
data_for_analysis.reset_index(inplace=True,drop=True)
print(data_for_analysis.info())

# risk_level is categorical, we need to convert it to numerical


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

# Examine explained variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print("Explained Variance by Component:")
for i in range(min(10, len(explained_variance_ratio))):
    print(f"PC{i+1}: {explained_variance_ratio[i]:.3f} ({explained_variance_ratio[i]*100:.1f}%)")

print(f"\nFirst 3 components explain {cumulative_variance[2]*100:.1f}% of total variance")
print(f"First 5 components explain {cumulative_variance[4]*100:.1f}% of total variance")

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
    pca.components_[:5].T,  # First 5 components
    columns=[f'PC{i+1}' for i in range(5)],
    index=data_for_analysis.columns
)

print("Principal Component Loadings (How much each variable contributes):")
print(components_df.round(3))

# Visualize component loadings for interpretation
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# PC1 loadings
pc1_loadings = components_df['PC1'].sort_values(key=abs, ascending=False)
axes[0,0].barh(range(len(pc1_loadings)), pc1_loadings.values)
axes[0,0].set_yticks(range(len(pc1_loadings)))
axes[0,0].set_yticklabels(pc1_loadings.index, fontsize=9)
axes[0,0].set_title(f'PC1 Loadings (Explains {explained_variance_ratio[0]*100:.1f}% of variance)')
axes[0,0].axvline(x=0, color='black', linestyle='-', alpha=0.3)

# PC2 loadings
pc2_loadings = components_df['PC2'].sort_values(key=abs, ascending=False)
axes[0,1].barh(range(len(pc2_loadings)), pc2_loadings.values, color='orange')
axes[0,1].set_yticks(range(len(pc2_loadings)))
axes[0,1].set_yticklabels(pc2_loadings.index, fontsize=9)
axes[0,1].set_title(f'PC2 Loadings (Explains {explained_variance_ratio[1]*100:.1f}% of variance)')
axes[0,1].axvline(x=0, color='black', linestyle='-', alpha=0.3)

# PC3 loadings
pc3_loadings = components_df['PC3'].sort_values(key=abs, ascending=False)
axes[1,0].barh(range(len(pc3_loadings)), pc3_loadings.values, color='green')
axes[1,0].set_yticks(range(len(pc3_loadings)))
axes[1,0].set_yticklabels(pc3_loadings.index, fontsize=9)
axes[1,0].set_title(f'PC3 Loadings (Explains {explained_variance_ratio[2]*100:.1f}% of variance)')
axes[1,0].axvline(x=0, color='black', linestyle='-', alpha=0.3)

# PC1 vs PC2 scatter plot of cities
axes[1,1].scatter(pca_results[:, 0], pca_results[:, 1], alpha=0.6)
axes[1,1].set_xlabel('PC1')
axes[1,1].set_ylabel('PC2')
axes[1,1].set_title('Cities in PC1-PC2 Space')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()