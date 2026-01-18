"""Inspect PBMC dataset structure and contents."""
import scvelo as scv
import os
import numpy as np
import anndata
import argparse
import random


def main():
    parser = argparse.ArgumentParser(description='Inspect PBMC dataset')
    parser.add_argument('--data', type=str, default='data/PBMC/pbmc68k.h5ad', help='Path to dataset')
    args = parser.parse_args()

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = args.data if os.path.isabs(args.data) else os.path.join(PROJECT_ROOT, args.data)

    if not os.path.exists(DATA_PATH):
        print(f"\nDataset not found at: {DATA_PATH}")
        print("Downloading raw PBMC68k dataset...")
        adata = scv.datasets.pbmc68k()
        RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data/PBMC/pbmc68k.h5ad")
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        adata.write(RAW_DATA_PATH)
        print(f"Saved to: {RAW_DATA_PATH}")
    else:
        print(f"Loading dataset from: {DATA_PATH}")
        adata = anndata.io.read_h5ad(DATA_PATH)

    print(f"\n{'='*80}")
    print(f"PBMC68K DATASET INSPECTION")
    print(f"{'='*80}")

    print(f"\nShape: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"Data type: {type(adata.X)}")
    print(f"Layers: {list(adata.layers.keys())}")

    print(f"\n{'='*80}")
    print(f"GENE NAMES")
    print(f"{'='*80}")
    print(f"\nFirst 50 genes:")
    for i, gene in enumerate(adata.var_names[:50]):
        print(f"  {i+1:3d}. {gene}")

    print(f"\nLast 20 genes:")
    for i, gene in enumerate(adata.var_names[-20:]):
        print(f"  {adata.shape[1]-20+i+1:3d}. {gene}")

    print(f"\nRandom sample of 30 genes:")
    random.seed(42)
    sample_indices = sorted(random.sample(range(len(adata.var_names)), 30))
    for idx in sample_indices:
        print(f"  {idx+1:5d}. {adata.var_names[idx]}")

    print(f"\n{'='*80}")
    print(f"GENE NAME PATTERNS")
    print(f"{'='*80}")
    gene_names_upper = adata.var_names.str.upper()

    print(f"\nPrefix patterns:")
    print(f"  MT-: {adata.var_names.str.startswith('MT-').sum()}")
    print(f"  RP: {adata.var_names.str.startswith('RP').sum()}")
    print(f"  RPS: {adata.var_names.str.startswith('RPS').sum()}")
    print(f"  RPL: {adata.var_names.str.startswith('RPL').sum()}")
    print(f"  OR: {adata.var_names.str.startswith('OR').sum()}")
    print(f"  LINC: {adata.var_names.str.startswith('LINC').sum()}")

    print(f"\n{'='*80}")
    print(f"COMMON MARKER GENES")
    print(f"{'='*80}")

    immune_markers = ['CD3D', 'CD3E', 'CD4', 'CD8A', 'CD8B', 'MS4A1', 'CD19', 'CD14',
                      'FCGR3A', 'CD68', 'NCAM1', 'IL7R', 'CCR7', 'NKG7', 'GNLY']

    print(f"\nImmune cell markers:")
    for marker in immune_markers:
        if marker.upper() in gene_names_upper.values:
            idx = list(gene_names_upper.values).index(marker.upper())
            print(f"  Found: {marker} -> {adata.var_names[idx]}")
        else:
            print(f"  Missing: {marker}")

    print(f"\n{'='*80}")
    print(f"EXPRESSION STATISTICS")
    print(f"{'='*80}")

    if hasattr(adata.X, 'toarray'):
        X_check = adata.X.toarray()
    else:
        X_check = np.array(adata.X)

    print(f"\nValue range: [{X_check.min():.4f}, {X_check.max():.4f}]")
    print(f"Mean: {X_check.mean():.4f}, Median: {np.median(X_check):.4f}")

    sparsity = (X_check == 0).sum() / X_check.size * 100
    print(f"Sparsity: {sparsity:.2f}% zeros")

    cells_per_gene = (X_check > 0).sum(axis=0)
    if len(cells_per_gene.shape) > 1:
        cells_per_gene = np.asarray(cells_per_gene).flatten()

    print(f"\nGenes by expression frequency:")
    for threshold in [1, 10, 100, 1000, 10000]:
        print(f"  >= {threshold} cells: {(cells_per_gene >= threshold).sum()}")

    print(f"\n{'='*80}")
    print(f"CELL METADATA")
    print(f"{'='*80}")
    print(f"\nColumns in adata.obs:")
    for col in adata.obs.columns:
        n_unique = adata.obs[col].nunique()
        print(f"  {col}: {adata.obs[col].dtype}, {n_unique} unique")
        if n_unique <= 20:
            print(f"    Values: {list(adata.obs[col].unique())}")

    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()
