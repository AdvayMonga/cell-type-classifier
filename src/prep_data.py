import scanpy as sc
import os

# Get the project root directory (parent of src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset configuration - save to project root
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data/PBMC/pbmc3k.h5ad")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data/PBMC/pbmc3k_processed.h5ad")
VIZ_DIR = os.path.join(PROJECT_ROOT, 'visualizations')

# Download raw data
os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

# Only download if the file doesn't exist
if not os.path.exists(RAW_DATA_PATH):
    print("Downloading dataset...")
    adata = sc.datasets.pbmc3k()
    adata.write(RAW_DATA_PATH)
else:
    print("Loading existing dataset...")
    adata = sc.read(RAW_DATA_PATH)

# Quality control filtering
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Normalization
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Feature selection
sc.pp.highly_variable_genes(adata, n_top_genes=500)
adata = adata[:, adata.var['highly_variable']].copy()

# Scaling
sc.pp.scale(adata, max_value=10)

# PCA for dimensionality reduction
sc.tl.pca(adata, n_comps=40)

# Clustering for cell type labels
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.leiden(adata, resolution=0.5, flavor ='igraph', n_iterations=2)

# Update config column name
adata.obs['louvain'] = adata.obs['leiden']
# Save processed data
adata.obs['cell_type'] = adata.obs["leiden"]
adata.write(PROCESSED_DATA_PATH)

print(f"Done: {adata.shape[0]} cells x {adata.shape[1]} genes")