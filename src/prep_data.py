import scanpy as sc
import os
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, N_TOP_GENES, CELL_TYPE_COLUMN

# Download raw data
adata = sc.datasets.pbmc3k()
os.makedirs("data", exist_ok=True)
adata.write(RAW_DATA_PATH)

# Quality control filtering
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Normalization
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Feature selection
sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES)
adata = adata[:, adata.var['highly_variable']]

# Scaling
sc.pp.scale(adata, max_value=10)

# PCA for dimensionality reduction
sc.tl.pca(adata, n_comps=40)

# Clustering for cell type labels
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.leiden(adata, resolution=0.5)

# Update config column name
adata.obs['louvain'] = adata.obs['leiden']
# Save processed data
adata.obs['cell_type'] = adata.obs[CELL_TYPE_COLUMN]
adata.write(PROCESSED_DATA_PATH)

print(f"Done: {adata.shape[0]} cells x {adata.shape[1]} genes")