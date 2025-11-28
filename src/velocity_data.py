import scvelo as scv
import os
import numpy as np
import anndata
import scanpy as sc
import matplotlib.pyplot as plt
import argparse

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
        adata = anndata.io.read_h5ad(RAW_DATA_PATH)

    # scv matplotlib settings
    scv.set_figure_params('scvelo')


    # Remove low-quality genes
    print("\nRemoving low-quality genes...")

    gene_names_upper = adata.var_names.str.upper()

    pseudogenes = gene_names_upper.str.match(r'^RP[^SL]')  # RP but not RPS or RPL
    olfactory = gene_names_upper.str.startswith('OR')
    mitochondrial = gene_names_upper.str.startswith('MT-')
    ribosomal = gene_names_upper.str.startswith(('RPS', 'RPL'))

    genes_to_remove = pseudogenes | olfactory | mitochondrial | ribosomal

    print(f"   Pseudogenes (RP*): {pseudogenes.sum()}")
    print(f"   Olfactory receptors (OR): {olfactory.sum()}")
    print(f"   Mitochondrial genes (MT-): {mitochondrial.sum()}")
    print(f"   Ribosomal genes (RPS, RPL): {ribosomal.sum()}")
    print(f"   Total genes to remove: {genes_to_remove.sum()}")

    adata = adata[:, ~genes_to_remove].copy()
    print(f"   After filtering: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Preprocessing
    print("\nPreprocessing data...")
    
    # Filter and normalize
    scv.pp.filter_genes(adata, min_cells=10)
    scv.pp.normalize_per_cell(adata)
    scv.pp.filter_genes_dispersion(adata, n_top_genes=3000)
    sc.pp.log1p(adata)

    
    # Compute moments (neighbors + pca)
    scv.pp.moments(adata, n_pcs=50, n_neighbors=30)
    print(f"   After preprocessing: {adata.shape[0]} cells x {adata.shape[1]} genes")

   # Estimate RNA velocity
    print("\nEstimating RNA velocity...")
    scv.tl.velocity(adata, mode='stochastic')
    print(f"   Velocity shape: {adata.layers['velocity'].shape}")

    # Velocity Graph
    scv.tl.velocity_graph(adata, n_jobs=1)
    print("   Velocity graph computed")

    # Compute UMAP (needed for velocity embedding plots)
    print("\nComputing UMAP...")
    sc.tl.umap(adata)
    print("   UMAP computed")

    # Project velocities onto UMAP
    if args.plot:
        scv.pl.velocity_embedding_stream(adata, basis='umap')
        plt.savefig(os.path.join(VIZ_DIR, 'velocity_stream.png'), dpi=300)
        plt.close()
        print(f"   Saved velocity stream plot")

    # Extract velocity features as scalars
    print("\nExtracting velocity features...")

    # Velocity magnitude (speed of expression change) - PER CELL
    velocity_matrix = adata.layers['velocity']
    if hasattr(velocity_matrix, 'toarray'):
        velocity_matrix = velocity_matrix.toarray()
    velocity_magnitude = np.sqrt((velocity_matrix ** 2).sum(axis=1))
    adata.obs['velocity_magnitude'] = velocity_magnitude
    print(f"   Added velocity_magnitude")

    # Velocity pseudotime (trajectory position) - PER CELL
    scv.tl.velocity_pseudotime(adata)
    print(f"   Added velocity_pseudotime")

    # Velocity confidence (certainty of estimate) - PER CELL
    scv.tl.velocity_confidence(adata)
    print(f"   Added velocity_confidence")

    # Latent time (developmental progression) - PER CELL
    scv.tl.recover_dynamics(adata, n_jobs=4)
    scv.tl.latent_time(adata)
    print(f"   Added latent_time")


    # Velocity Confidence Visualization
    if args.plot:
        keys = 'velocity_length', 'velocity_confidence'
        scv.pl.scatter(adata, c=keys, cmap='coolwarm', perc=[5, 95])
        plt.savefig(os.path.join(VIZ_DIR, 'velocity_confidence.png'), dpi=300)
        plt.close()
        print(f"   Saved velocity confidence plot")

    # Pseudotime Visualization
    if args.plot:
        scv.pl.scatter(adata, color='velocity_pseudotime', cmap='gnuplot')
        plt.savefig(os.path.join(VIZ_DIR, 'velocity_pseudotime.png'), dpi=300)
        plt.close()
        print(f"   Saved velocity pseudotime plot")
    
    

    # Print Summary
    print(f"\n   Velocity features added to adata.obs:")
    print(f"     - velocity_magnitude: shape {adata.obs['velocity_magnitude'].shape}")
    print(f"     - velocity_pseudotime: shape {adata.obs['velocity_pseudotime'].shape}")
    print(f"     - velocity_confidence: shape {adata.obs['velocity_confidence'].shape}")
    print(f"     - latent_time: shape {adata.obs['latent_time'].shape}")

    # Assign Cell Types
    print("\n6. Assigning cell types...")
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


    # Save Processed Data
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

    if args.plot:
        print(f"Visualizations saved to: visualizations/")

    pass

if __name__ == '__main__':
    main()