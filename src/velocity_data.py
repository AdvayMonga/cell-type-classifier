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

    # DIAGNOSTIC: Check what we're starting with
    print(f"\n{'='*60}")
    print(f"INITIAL DATASET DIAGNOSTICS")
    print(f"{'='*60}")
    print(f"Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"Layers available: {list(adata.layers.keys())}")
    print(f"Data type: {type(adata.X)}")

    # Check gene expression distribution
    if hasattr(adata.X, 'toarray'):
        X_check = adata.X.toarray()
    else:
        X_check = np.array(adata.X)

    # Count cells expressing each gene
    cells_per_gene = (X_check > 0).sum(axis=0)
    if len(cells_per_gene.shape) > 1:
        cells_per_gene = np.asarray(cells_per_gene).flatten()

    print(f"\nGene expression statistics:")
    print(f"  Genes expressed in ≥1 cells: {(cells_per_gene >= 1).sum()}")
    print(f"  Genes expressed in ≥3 cells: {(cells_per_gene >= 3).sum()}")
    print(f"  Genes expressed in ≥10 cells: {(cells_per_gene >= 10).sum()}")
    print(f"  Genes expressed in ≥20 cells: {(cells_per_gene >= 20).sum()}")
    print(f"  Genes expressed in ≥50 cells: {(cells_per_gene >= 50).sum()}")
    print(f"  Genes expressed in ≥100 cells: {(cells_per_gene >= 100).sum()}")
    print(f"  Genes expressed in ≥1000 cells: {(cells_per_gene >= 1000).sum()}")

    print(f"\nGene expression distribution:")
    print(f"  Median cells per gene: {np.median(cells_per_gene):.0f}")
    print(f"  Mean cells per gene: {np.mean(cells_per_gene):.0f}")
    print(f"  Min cells per gene: {np.min(cells_per_gene):.0f}")
    print(f"  Max cells per gene: {np.max(cells_per_gene):.0f}")

    # Check if data is already log-normalized
    print(f"\nData value range check:")
    print(f"  Min value: {X_check.min():.4f}")
    print(f"  Max value: {X_check.max():.4f}")
    print(f"  Mean value: {X_check.mean():.4f}")
    print(f"  Median value: {np.median(X_check):.4f}")
    
    if X_check.max() > 100:
        print(f"  ⚠️  Data appears to be RAW COUNTS (not normalized)")
    elif X_check.max() > 10:
        print(f"  ⚠️  Data might be normalized but not log-transformed")
    else:
        print(f"  ✓ Data appears to be log-normalized")

    # Check sparsity
    sparsity = (X_check == 0).sum() / X_check.size * 100
    print(f"\nData sparsity: {sparsity:.1f}% zeros")
    
    print(f"{'='*60}\n")

    # Remove low-quality genes BUT preserve cell cycle genes
    print("\nRemoving low-quality genes...")

    gene_names_upper = adata.var_names.str.upper()

    # Define cell cycle genes to ALWAYS keep
    s_genes = ['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6', 
               'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP', 'HELLS', 'RFC2', 'RPA2', 'NASP', 
               'RAD51AP1', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 
               'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 
               'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8']

    g2m_genes = ['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 
                 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'FAM64A', 'SMC4', 'CCNB2', 
                 'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 
                 'KIF20B', 'HJURP', 'CDCA3', 'HN1', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 
                 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1', 
                 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA']

    cell_cycle_genes_upper = set([g.upper() for g in s_genes + g2m_genes])
    is_cell_cycle_gene = gene_names_upper.isin(cell_cycle_genes_upper)

    pseudogenes = gene_names_upper.str.match(r'^RP[^SL]')  # RP but not RPS or RPL
    olfactory = gene_names_upper.str.startswith('OR')
    mitochondrial = gene_names_upper.str.startswith('MT-')
    ribosomal = gene_names_upper.str.startswith(('RPS', 'RPL'))

    # Don't remove cell cycle genes even if they match patterns
    genes_to_remove = (pseudogenes | olfactory | mitochondrial | ribosomal) & ~is_cell_cycle_gene

    print(f"   Pseudogenes (RP*): {pseudogenes.sum()}")
    print(f"   Olfactory receptors (OR): {olfactory.sum()}")
    print(f"   Mitochondrial genes (MT-): {mitochondrial.sum()}")
    print(f"   Ribosomal genes (RPS, RPL): {ribosomal.sum()}")
    print(f"   Cell cycle genes in dataset: {is_cell_cycle_gene.sum()}")
    print(f"   Total genes to remove: {genes_to_remove.sum()}")

    adata = adata[:, ~genes_to_remove].copy()
    print(f"   After filtering: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Preprocessing
    print("\nPreprocessing data...")
    
    # Filter and normalize - keep all genes that pass basic quality filter
    scv.pp.filter_genes(adata, min_shared_counts=10)
    print(f"   ✓ After min_shared_counts filter: {adata.n_vars} genes")
    
    scv.pp.normalize_per_cell(adata)
    print(f"   ✓ Normalized per cell")
    
    # SKIPPING dispersion filtering to keep all ~10k+ quality genes
    # instead of filtering down to top 3000 variable genes
    print(f"   ✓ Skipping dispersion filter - keeping all {adata.n_vars} genes")
    
    # Check cell cycle gene retention
    cell_cycle_mask = adata.var_names.str.upper().isin(cell_cycle_genes_upper)
    print(f"   ✓ Cell cycle genes present: {cell_cycle_mask.sum()}")
    
    # Log transform BEFORE computing neighbors (critical for proper neighbor graph)
    sc.pp.log1p(adata)
    print(f"   ✓ Log-transformed data")
    print(f"   Final gene count: {adata.n_vars} genes")

    # Now compute PCA, neighbors, and moments on log-transformed data
    print("\n   Computing PCA, neighbors, and moments...")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=False)
    sc.tl.pca(adata, n_comps=50)
    print(f"   ✓ PCA computed (50 components)")

    # Compute neighbors in PCA space (on log-transformed data)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
    print(f"   ✓ Neighbors computed (30 PCs, 30 neighbors)")

    # Compute moments using pre-computed neighbors
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)  # Use pre-computed
    print(f"   ✓ Moments computed")
    print(f"   Final shape: {adata.shape[0]} cells x {adata.shape[1]} genes")

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

    # 1. Velocity pseudotime (trajectory position) - PER CELL
    scv.tl.velocity_pseudotime(adata)
    print(f"   Added velocity_pseudotime (mean: {adata.obs['velocity_pseudotime'].mean():.4f})")

    # 2. Latent time (developmental progression) - PER CELL
    try:
        scv.tl.recover_dynamics(adata, n_jobs=4)
        scv.tl.latent_time(adata)
        print(f"   Added latent_time (mean: {adata.obs['latent_time'].mean():.4f})")
    except Exception as e:
        print(f"   Warning: Could not compute latent_time: {e}")
        adata.obs['latent_time'] = adata.obs['velocity_pseudotime'].copy()

    # 3. Velocity confidence (certainty of velocity estimate) - PER CELL
    scv.tl.velocity_confidence(adata)
    print(f"   Added velocity_confidence (mean: {adata.obs['velocity_confidence'].mean():.4f})")

    # 4. Velocity magnitude (speed of transcriptional change) - PER CELL
    velocity_matrix = adata.layers['velocity']
    if hasattr(velocity_matrix, 'toarray'):
        velocity_matrix = velocity_matrix.toarray()
    velocity_magnitude = np.sqrt((velocity_matrix ** 2).sum(axis=1))
    adata.obs['velocity_magnitude'] = velocity_magnitude
    print(f"   Added velocity_magnitude (mean: {velocity_magnitude.mean():.4f})")

    # 5. Cell cycle phase scores (helps distinguish proliferating cells)
    print("\nComputing cell cycle scores...")
    try:
        # Check which cell cycle genes are actually present
        s_genes_present = [g for g in s_genes if g.upper() in adata.var_names.str.upper().values]
        g2m_genes_present = [g for g in g2m_genes if g.upper() in adata.var_names.str.upper().values]
        
        print(f"   S-phase genes found: {len(s_genes_present)}/{len(s_genes)}")
        print(f"   G2M-phase genes found: {len(g2m_genes_present)}/{len(g2m_genes)}")
        
        if len(s_genes_present) >= 5 and len(g2m_genes_present) >= 5:
            scv.tl.score_genes_cell_cycle(adata)
            print(f"   ✓ Added S_score (mean: {adata.obs['S_score'].mean():.4f})")
            print(f"   ✓ Added G2M_score (mean: {adata.obs['G2M_score'].mean():.4f})")
        else:
            raise ValueError(f"Not enough cell cycle genes ({len(s_genes_present)} S-phase, {len(g2m_genes_present)} G2M)")
            
    except (ValueError, Exception) as e:
        print(f"   ⚠ Warning: {e}")
        print(f"   Using expression statistics as proxy for cell cycle activity")
        
        if hasattr(adata.X, 'toarray'):
            X_dense = adata.X.toarray()
        else:
            X_dense = adata.X
        
        # Normalize to 0-1 range
        mean_expr = X_dense.mean(axis=1)
        expr_var = X_dense.var(axis=1)
        
        adata.obs['S_score'] = (mean_expr - mean_expr.min()) / (mean_expr.max() - mean_expr.min())
        adata.obs['G2M_score'] = (expr_var - expr_var.min()) / (expr_var.max() - expr_var.min())
        
        print(f"   Added S_score proxy (mean: {adata.obs['S_score'].mean():.4f})")
        print(f"   Added G2M_score proxy (mean: {adata.obs['G2M_score'].mean():.4f})")

    # 6. Gene expression statistics (transcriptional complexity)
    print("\nComputing expression statistics...")
    if hasattr(adata.X, 'toarray'):
        X_dense = adata.X.toarray()
    else:
        X_dense = adata.X
    
    adata.obs['mean_expression'] = X_dense.mean(axis=1)
    adata.obs['expression_variance'] = X_dense.var(axis=1)
    adata.obs['n_genes_expressed'] = (X_dense > 0).sum(axis=1)
    
    print(f"   Added mean_expression (mean: {adata.obs['mean_expression'].mean():.4f})")
    print(f"   Added expression_variance (mean: {adata.obs['expression_variance'].mean():.4f})")
    print(f"   Added n_genes_expressed (mean: {adata.obs['n_genes_expressed'].mean():.1f})")

    # Visualizations
    if args.plot:
        # Pseudotime Visualization
        scv.pl.scatter(adata, color='velocity_pseudotime', cmap='gnuplot')
        plt.savefig(os.path.join(VIZ_DIR, 'velocity_pseudotime.png'), dpi=300)
        plt.close()
        print(f"   Saved velocity pseudotime plot")
        
        # Latent time visualization (if computed successfully)
        if 'latent_time' in adata.obs.columns and not np.all(adata.obs['latent_time'] == 0.0):
            scv.pl.scatter(adata, color='latent_time', cmap='viridis')
            plt.savefig(os.path.join(VIZ_DIR, 'latent_time.png'), dpi=300)
            plt.close()
            print(f"   Saved latent time plot")
    

    # Print Summary
    print(f"\n   All features added to adata.obs:")
    feature_cols = ['velocity_pseudotime', 'latent_time', 'velocity_confidence', 'velocity_magnitude',
                    'S_score', 'G2M_score', 'mean_expression', 'expression_variance', 'n_genes_expressed']
    for feat in feature_cols:
        if feat in adata.obs.columns:
            print(f"     - {feat}: shape {adata.obs[feat].shape}")

    # Assign Cell Types
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