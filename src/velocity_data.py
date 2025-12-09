"""
Main preprocessing pipeline for PBMC data with RNA velocity.

This script orchestrates the preprocessing modules to:
1. Load raw PBMC68k data
2. Filter low-quality genes (while protecting critical markers)
3. Normalize and log-transform
4. Save unsmoothed data for classification
5. Compute RNA velocity
6. Extract velocity and expression features
7. Normalize and boost features
8. Save processed data

Usage:
    python velocity_data_refactored.py
    python velocity_data_refactored.py --plot  # Generate plots
"""
import scvelo as scv
import scanpy as sc
import os
import argparse

# Import preprocessing modules
from preprocessing import (
    filter_low_quality_genes,
    compute_velocity_features,
    extract_velocity_features,
    normalize_and_boost_features
)
from utils import check_data_quality


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Preprocess PBMC data with scVelo')
    parser.add_argument('--plot', action='store_true', help='Generate and save plots')
    args = parser.parse_args()

    # Dataset configuration
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data/PBMC/pbmc68k.h5ad")
    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data/PBMC/pbmc68k_processed.h5ad")
    VIZ_DIR = os.path.join(PROJECT_ROOT, 'visualizations')

    # Create directories
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    # Download or load dataset
    if not os.path.exists(RAW_DATA_PATH):
        print("\nDownloading dataset...")
        adata = scv.datasets.pbmc68k()
        adata.write(RAW_DATA_PATH)
    else:
        print("Loading existing dataset...")
        import anndata
        adata = anndata.io.read_h5ad(RAW_DATA_PATH)

    # Configure scvelo
    scv.set_figure_params('scvelo')

    # Step 1: Check initial data quality
    print(f"\n{'='*60}")
    print(f"INITIAL DATASET DIAGNOSTICS")
    print(f"{'='*60}")
    check_data_quality(adata, verbose=True)

    # Step 2: Filter low-quality genes (preserving critical markers)
    adata, protected = filter_low_quality_genes(
        adata, 
        min_cells=10,
        remove_mt=True,
        remove_ribo=True,
        remove_pseudogenes=True,
        remove_olfactory=True
    )

    # Step 3: Basic preprocessing
    print("\nPreprocessing data...")
    scv.pp.filter_genes(adata, min_cells=10)
    print(f"   ✓ After min_cells filter: {adata.n_vars} genes")
    
    scv.pp.normalize_per_cell(adata)
    print(f"   ✓ Normalized per cell")
    
    # Check protected gene retention
    protected_genes_upper = set([g.upper() for g in 
                                protected['s_phase'] + protected['g2m_phase'] + protected['markers']])
    protected_mask = adata.var_names.str.upper().isin(protected_genes_upper)
    print(f"   ✓ Protected genes retained: {protected_mask.sum()}/{len(protected_genes_upper)}")
    
    # Log transform
    sc.pp.log1p(adata)
    print(f"   ✓ Log-transformed data")
    print(f"   Final gene count: {adata.n_vars} genes")

    # Step 4: Compute velocity (this also saves unsmoothed data)
    adata = compute_velocity_features(adata, n_pcs=30, n_neighbors=30)

    # Step 5: Extract velocity and expression features
    adata = extract_velocity_features(adata, protected)

    # Step 6: Normalize and boost features
    adata = normalize_and_boost_features(adata, boost_factor=1.5)

    # Step 7: Visualizations (optional)
    if args.plot:
        print("\nGenerating visualizations...")
        import matplotlib.pyplot as plt
        
        scv.pl.velocity_embedding_stream(adata, basis='umap')
        plt.savefig(os.path.join(VIZ_DIR, 'velocity_stream.png'), dpi=300)
        plt.close()
        print(f"   ✓ Saved velocity stream plot")
        
        scv.pl.scatter(adata, color='velocity_pseudotime', cmap='gnuplot')
        plt.savefig(os.path.join(VIZ_DIR, 'velocity_pseudotime.png'), dpi=300)
        plt.close()
        print(f"   ✓ Saved velocity pseudotime plot")
        
        if 'latent_time' in adata.obs.columns and not (adata.obs['latent_time'] == 0.0).all():
            scv.pl.scatter(adata, color='latent_time', cmap='viridis')
            plt.savefig(os.path.join(VIZ_DIR, 'latent_time.png'), dpi=300)
            plt.close()
            print(f"   ✓ Saved latent time plot")

    # Step 8: Assign cell types
    print("\nAssigning cell types...")
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

    # Step 9: Save processed data
    print("\nSaving processed data...")
    adata.write(PROCESSED_DATA_PATH)

    # Final summary
    print(f"\n{'='*60}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Final dataset: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"Unique cell types: {adata.obs['cell_type'].nunique()}")
    print(f"\nCell type distribution:")
    print(adata.obs['cell_type'].value_counts())
    print(f"\nData saved to: {PROCESSED_DATA_PATH}")
    print(f"Layers available: {list(adata.layers.keys())}")
    print(f"  ✓ 'unsmoothed_log': Unsmoothed expression for classification")
    print(f"  ✓ adata.X: Smoothed expression for velocity")
    
    if args.plot:
        print(f"\nVisualizations saved to: {VIZ_DIR}/")


if __name__ == '__main__':
    main()
