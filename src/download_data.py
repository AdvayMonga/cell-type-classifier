import scanpy as sc
import os

print("Downloading datasets...")

adata = sc.datasets.pbmc3k_processed()
cells = adata.shape[0]
genes = adata.shape[1]

print(f"Dataset: {cells} cells x {genes} genes")

# louvain column with cell type labels
# prints how many cell types are present
print(adata.obs['louvain'].value_counts())

# creates data folder
os.makedirs("data", exist_ok=True)
adata.write("data/pbmc3k_processed.h5ad")
