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

    # Remove low-quality genes BUT preserve critical markers
    print("\nRemoving low-quality genes...")

    gene_names_upper = adata.var_names.str.upper()

    # Define S-phase cell cycle genes to ALWAYS keep
    s_genes = ['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6', 
               'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'HELLS', 'RFC2', 'RPA2', 'NASP', 
               'RAD51AP1', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 
               'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 
               'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8']

    # Define G2M-phase cell cycle genes to ALWAYS keep
    g2m_genes = ['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 
                 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'SMC4', 'CCNB2', 
                 'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 
                 'KIF20B', 'HJURP', 'CDCA3', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 
                 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1', 
                 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA']

    # CRITICAL: Cell surface markers from Zheng et al. 2017 Nature Communications
    # These are the markers used for FACS sorting and are essential for classification
    cell_surface_markers = [
        # T cell markers (CD3 complex)
        'CD3D', 'CD3E', 'CD3G',          # Pan T cell markers
        'CD4',                            # T helper cells
        'CD8A', 'CD8B',                  # Cytotoxic T cells
        'IL7R',                           # CD127 - Naive T cells
        'CCR7',                           # Naive T cell homing
        'CD27',                           # Memory T cells
        'SELL',                           # L-selectin (CD62L) - Naive
        
        # B cell markers
        'CD19',                           # Pan B cell
        'MS4A1',                          # CD20
        'CD79A', 'CD79B',                # B cell receptor components
        
        # NK cell markers
        'NCAM1',                          # CD56 - NK cells
        'NKG7', 'GNLY',                  # NK cytotoxicity
        'KLRD1', 'KLRB1',                # Killer cell receptors
        
        # Monocyte markers
        'CD14',                           # Classical monocytes
        'FCGR3A',                         # CD16 - Non-classical monocytes
        'S100A8', 'S100A9',              # Monocyte-specific calgranulins
        'LYZ',                            # Lysozyme
        'CD68',                           # Macrophage marker
        
        # Dendritic cell markers
        'FCER1A',                         # Classical DC
        'CD1C',                           # cDC2
        
        # Stem cell markers
        'CD34',                           # Hematopoietic stem cells
        
        # Pan-leukocyte
        'PTPRC',                          # CD45 - All leukocytes
    ]

    # Combine all protected genes
    immune_markers = cell_surface_markers  # For backward compatibility
    print(f"   Protected gene categories:")
    print(f"     - S-phase cell cycle: {len(s_genes)} genes")
    print(f"     - G2M-phase cell cycle: {len(g2m_genes)} genes")
    print(f"     - Cell surface markers (Zheng 2017): {len(cell_surface_markers)} genes")

    # Combine all protected genes
    protected_genes = s_genes + g2m_genes + immune_markers
    protected_genes_upper = set([g.upper() for g in protected_genes])
    is_protected_gene = gene_names_upper.isin(protected_genes_upper)

    # Identify junk genes to remove
    pseudogenes = gene_names_upper.str.match(r'^RP[^SL]')  # RP but not RPS or RPL
    olfactory = gene_names_upper.str.startswith('OR')
    mitochondrial = gene_names_upper.str.startswith('MT-')
    ribosomal = gene_names_upper.str.startswith(('RPS', 'RPL'))

    # NEVER remove protected genes, even if they match removal patterns
    genes_to_remove = (pseudogenes | olfactory | mitochondrial | ribosomal) & ~is_protected_gene

    print(f"   Pseudogenes (RP*): {pseudogenes.sum()}")
    print(f"   Olfactory receptors (OR*): {olfactory.sum()}")
    print(f"   Mitochondrial genes (MT-*): {mitochondrial.sum()}")
    print(f"   Ribosomal genes (RPS*, RPL*): {ribosomal.sum()}")
    print(f"   Protected genes found: {is_protected_gene.sum()} ({len(s_genes)} S-phase + {len(g2m_genes)} G2M + {len(immune_markers)} immune)")
    print(f"   Total genes to remove: {genes_to_remove.sum()}")

    adata = adata[:, ~genes_to_remove].copy()
    print(f"   ✓ After filtering: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Preprocessing
    print("\nPreprocessing data...")
    
    # Filter and normalize - keep all genes that pass basic quality filter
    scv.pp.filter_genes(adata, min_cells=10)
    print(f"   ✓ After min_cells filter: {adata.n_vars} genes")
    
    scv.pp.normalize_per_cell(adata)
    print(f"   ✓ Normalized per cell")

    print(f"   ✓ Skipping dispersion filter")

    # Check protected gene retention
    protected_mask = adata.var_names.str.upper().isin(protected_genes_upper)
    print(f"   ✓ Protected genes retained: {protected_mask.sum()}/{len(protected_genes)}")
    
    # Log transform BEFORE computing neighbors (critical for proper neighbor graph)
    sc.pp.log1p(adata)
    print(f"   ✓ Log-transformed data")
    print(f"   Final gene count: {adata.n_vars} genes")
    
    # CRITICAL: Save unsmoothed log-transformed data for classification
    # The moments() function will smooth adata.X across neighbors, which helps velocity
    # estimation but hurts classification by blurring cell type boundaries
    adata.layers['unsmoothed_log'] = adata.X.copy()
    print(f"   ✓ Saved unsmoothed log-transformed data to layer 'unsmoothed_log' for classification")
    print(f"      (adata.X will be smoothed by moments, but unsmoothed version preserved)")

    # Now compute PCA, neighbors, and moments on log-transformed data
    print("\n   Computing PCA, neighbors, and moments...")
    
    # DO NOT select genes here - keep all genes for flexibility
    # Gene selection will be done in select_genes.py as a separate fast step
    print(f"   ✓ Keeping all {adata.n_vars} genes (no selection)")
    
    sc.tl.pca(adata, n_comps=50)
    print(f"   ✓ PCA computed (50 components)")

    # Compute neighbors in PCA space (on log-transformed data)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
    print(f"   ✓ Neighbors computed (30 PCs, 30 neighbors)")

    # Compute moments using pre-computed neighbors
    # This smooths adata.X across neighbors (good for velocity, preserved unsmoothed for classification)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)  # Use pre-computed
    print(f"   ✓ Moments computed (adata.X is now smoothed for velocity estimation)")
    print(f"   ✓ Unsmoothed data available in layer 'unsmoothed_log' for classification")
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

    # Get velocity matrix for feature computation
    velocity_matrix = adata.layers['velocity']
    if hasattr(velocity_matrix, 'toarray'):
        velocity_matrix = velocity_matrix.toarray()
    
    # Compute velocity magnitude (L2 norm)
    velocity_magnitude = np.sqrt((velocity_matrix ** 2).sum(axis=1))
    
    # 1. Velocity pseudotime - normalized velocity magnitude
    # (Skipping expensive scv.tl.velocity_pseudotime which takes 15+ min)
    adata.obs['velocity_pseudotime'] = (velocity_magnitude - velocity_magnitude.min()) / \
                                        (velocity_magnitude.max() - velocity_magnitude.min())
    print(f"   ✓ Added velocity_pseudotime (mean: {adata.obs['velocity_pseudotime'].mean():.4f})")

    # 2. Latent time - use velocity direction consistency as proxy
    # (Skipping scv.tl.latent_time which recomputes neighbors and is very slow)
    # Compute per-gene velocity consistency (signed velocity magnitude)
    velocity_direction = np.sign(velocity_matrix).mean(axis=1)
    
    # Ensure it's a flat 1D array, not a matrix
    if hasattr(velocity_direction, 'A1'):
        velocity_direction = velocity_direction.A1  # Sparse matrix to flat array
    else:
        velocity_direction = np.asarray(velocity_direction).flatten()
    
    # Normalize to 0-1 range
    vd_min = velocity_direction.min()
    vd_max = velocity_direction.max()
    vd_range = vd_max - vd_min
    
    if vd_range < 1e-10:
        # All velocities are the same direction - set to 0.5 (neutral)
        print(f"   ⚠️  All velocity directions identical, setting latent_time to 0.5")
        adata.obs['latent_time'] = np.full(len(velocity_direction), 0.5)
    else:
        adata.obs['latent_time'] = (velocity_direction - vd_min) / vd_range
    
    print(f"   ✓ Added latent_time proxy (mean: {adata.obs['latent_time'].mean():.4f})")

    # 3. Velocity confidence (certainty of velocity estimate) - PER CELL
    scv.tl.velocity_confidence(adata)
    print(f"   ✓ Added velocity_confidence (mean: {adata.obs['velocity_confidence'].mean():.4f})")

    # 4. Velocity magnitude (raw magnitude values)
    adata.obs['velocity_magnitude'] = velocity_magnitude
    print(f"   ✓ Added velocity_magnitude (mean: {velocity_magnitude.mean():.4f})")

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
    
    # Compute statistics and ensure they're 1D arrays
    mean_expr = np.asarray(X_dense.mean(axis=1)).flatten()
    expr_var = np.asarray(X_dense.var(axis=1)).flatten()
    n_genes = np.asarray((X_dense > 0).sum(axis=1)).flatten()
    
    adata.obs['mean_expression'] = mean_expr
    adata.obs['expression_variance'] = expr_var
    adata.obs['n_genes_expressed'] = n_genes
    
    print(f"   Added mean_expression (mean: {mean_expr.mean():.4f})")
    print(f"   Added expression_variance (mean: {expr_var.mean():.4f})")
    print(f"   Added n_genes_expressed (mean: {n_genes.mean():.1f})")

    # NORMALIZE ALL FEATURES TO 0-1 RANGE, then BOOST by 1.5x
    print("\nNormalizing and boosting velocity features...")
    velocity_features = ['velocity_pseudotime', 'latent_time', 'velocity_confidence', 
                         'velocity_magnitude', 'S_score', 'G2M_score', 
                         'mean_expression', 'expression_variance', 'n_genes_expressed']
    
    velocity_boost_factor = 1.5
    print(f"   Boost factor: {velocity_boost_factor}x (to balance with gene expression 0-4.38 range)")
    
    for feat in velocity_features:
        if feat in adata.obs.columns:
            values = adata.obs[feat].values
            
            # Check for NaNs BEFORE normalization (shouldn't exist!)
            nan_count = np.isnan(values).sum()
            if nan_count > 0:
                print(f"   ❌ ERROR: {feat} contains {nan_count} NaNs BEFORE normalization!")
                print(f"      This indicates a bug in the computation. Replacing with 0...")
                values = np.nan_to_num(values, nan=0.0)
            
            # Normalize to 0-1 range
            val_min = values.min()
            val_max = values.max()
            val_range = val_max - val_min
            
            if val_range < 1e-10:
                # All values are identical
                print(f"   ⚠️  {feat} has no variance (constant: {val_min:.4f}), setting to 0.5")
                normalized = np.full_like(values, 0.5)
            else:
                normalized = (values - val_min) / val_range
            
            # Check for NaNs AFTER normalization (shouldn't happen if fixed above)
            nan_count_after = np.isnan(normalized).sum()
            if nan_count_after > 0:
                print(f"   ❌ ERROR: {feat} produced {nan_count_after} NaNs AFTER normalization!")
                print(f"      Replacing with 0.0 as fallback")
                normalized = np.nan_to_num(normalized, nan=0.0)
            
            # BOOST by 1.5x to increase influence relative to genes
            boosted = normalized * velocity_boost_factor
            
            adata.obs[feat] = boosted
            print(f"   ✓ Normalized & boosted {feat}: range [{boosted.min():.4f}, {boosted.max():.4f}], mean={boosted.mean():.4f}")

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