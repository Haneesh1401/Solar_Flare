import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data # Features
y = iris.target # True labels (for visualization)

# --- Step 1: Standardize the data ---
# PCA is affected by scale, so it's common practice to standardize the data.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Step 2: Perform PCA ---
# Create a PCA object, specifying the number of components to keep (e.g., 2)
n_components = 2
pca = PCA(n_components=n_components)

# Fit PCA on the scaled data
pca.fit(X_scaled)

# Transform the data to the new component space
X_pca = pca.transform(X_scaled)

print(f"Original Data Shape: {X.shape}")
print(f"Transformed Data Shape (PCA): {X_pca.shape}")

# --- Step 3: Analyze and Visualize PCA results ---
# Explained Variance Ratio: The amount of variance explained by each selected component
print("\nExplained Variance Ratio:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"Principal Component {i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

print(f"\nCumulative Explained Variance: {pca.explained_variance_ratio_.sum():.4f}")

# Visualize the 2D data after PCA
plt.figure(figsize=(8, 6))
# Scatter plot the data, colored by the original target class
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'2-Component PCA of Iris Dataset (Total Variance Explained: {pca.explained_variance_ratio_.sum()*100:.2f}%)')
handles, _ = scatter.legend_elements()
plt.legend(handles, iris.target_names, title="Classes")
plt.grid(True)
plt.show()