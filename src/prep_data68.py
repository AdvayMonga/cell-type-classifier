import scanpy as sc
import scvelo as scv
import os
import numpy as np
import matplotlib.pyplot as plt

# Get the project root directory (parent of src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset configuration - save to project root
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data/PBMC/pbmc68k_processed.h5ad")
VIZ_DIR = os.path.join(PROJECT_ROOT, 'visualizations')

os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

# Always download fresh from scvelo (don't cache the raw data)
print("Downloading dataset...")
adata = scv.datasets.pbmc68k()

print(f"Original data: {adata.shape[0]} cells x {adata.shape[1]} genes")
print(f"Available columns: {adata.obs.columns.tolist()}")

# QUALITY CONTROL - Calculate metrics and visualize
print("\n Quality Control...")
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

# DEBUG: Check what the gene counts actually are
print(f"   Min genes per cell: {adata.obs['n_genes_by_counts'].min()}")
print(f"   Max genes per cell: {adata.obs['n_genes_by_counts'].max()}")
print(f"   Mean genes per cell: {adata.obs['n_genes_by_counts'].mean():.0f}")
print(f"   Cells with >= 200 genes: {(adata.obs['n_genes_by_counts'] >= 200).sum()}")

# REMOVE MITOCHONDRIAL GENES
print("\n   Removing mitochondrial genes...")
# Identify mitochondrial genes (start with MT-)
mt_genes = adata.var_names.str.upper().str.startswith('MT-')
print(f"   Found {mt_genes.sum()} mitochondrial genes")
adata = adata[:, ~mt_genes].copy()
print(f"   After removing MT genes: {adata.shape[0]} cells x {adata.shape[1]} genes")


# Plot QC metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(adata.obs['n_genes_by_counts'], bins=100)
axes[0].set_xlabel('Number of genes')
axes[0].set_ylabel('Number of cells')
axes[0].set_title('Genes per cell')

axes[1].hist(adata.obs['total_counts'], bins=100)
axes[1].set_xlabel('Total counts')
axes[1].set_ylabel('Number of cells')
axes[1].set_title('UMI counts per cell')

axes[2].scatter(adata.obs['total_counts'], adata.obs['n_genes_by_counts'], alpha=0.1)
axes[2].set_xlabel('Total counts')
axes[2].set_ylabel('Number of genes')
axes[2].set_title('UMI vs Genes')
plt.tight_layout()
plt.savefig('visualizations/qc_metrics.png', dpi=300)
print("   Saved QC metrics plot")

# NORMALIZATION & FILTERING - Keep sparse format throughout
print("\n Normalization and filtering...")
print(f"   Before filtering: {adata.shape[0]} cells x {adata.shape[1]} genes")

# Filter cells: keep cells with at least x genes
scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=5000)
print(f"    After filtering cells: {adata.shape[0]} cells x {adata.shape[1]} genes")


# Scale the data
sc.pp.scale(adata, max_value=10)

# convert to dense for the models (after all preprocessing)
if hasattr(adata.X, 'toarray'):
    print("Converting to dense array for model training...")
    adata.X = adata.X.toarray()

# PCA ANALYSIS
print("\n PCA Analysis...")
sc.tl.pca(adata, n_comps=50)

# Plot variance ratio
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
axes[0].plot(range(1, 51), adata.uns['pca']['variance_ratio'])
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Explained')
axes[0].set_title('Scree Plot - PCA Variance Explained')
axes[0].grid(True)

# Cumulative variance
cumsum = np.cumsum(adata.uns['pca']['variance_ratio'])
axes[1].plot(range(1, 51), cumsum)
axes[1].axhline(y=0.85, color='r', linestyle='--', label='85% variance')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance Explained')
axes[1].set_title('Cumulative Variance Explained')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('visualizations/pca_analysis.png', dpi=300)
print("   Saved PCA analysis plot")

# Print cumulative variance
print(f"\n   Cumulative variance explained:")
for i in [10, 20, 30, 40, 50]:
    print(f"     {i} PCs: {cumsum[i-1]:.1%}")


# NEIGHBORHOOD GRAPH & CLUSTERING
print("\n4. Computing neighborhood graph...")
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=50)

# UMAP for visualization
# print("\n Computing UMAP...")
# sc.tl.umap(adata)

# ASSIGN CELL TYPES
print("\n Assigning cell types...")
if 'celltype' in adata.obs.columns:
    adata.obs['cell_type'] = adata.obs['celltype']
    print(f"   Using existing 'celltype' annotations")
elif 'clusters' in adata.obs.columns:
    adata.obs['cell_type'] = adata.obs['clusters']
    print(f"   Using existing 'clusters' annotations")
else:
    sc.tl.leiden(adata, flavor='igraph', resolution=0.5)
    adata.obs['cell_type'] = adata.obs['leiden']
    print(f"   Computed Leiden clustering")

# SAVE PROCESSED DATA
print("\n Saving processed data...")
adata.write(PROCESSED_DATA_PATH)

print(f"\n{'='*60}")
print(f"PREPROCESSING COMPLETE")
print(f"{'='*60}")
print(f"Final dataset: {adata.shape[0]} cells x {adata.shape[1]} genes")
print(f"Unique cell types: {adata.obs['cell_type'].nunique()}")
print(f"\nCell type distribution:")
print(adata.obs['cell_type'].value_counts())
print(f"\nData saved to: {PROCESSED_DATA_PATH}")
print(f"Visualizations saved to: visualizations/")