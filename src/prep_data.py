"""
Main preprocessing pipeline for PBMC data with RNA velocity.

This script performs the complete preprocessing workflow:
1. Load raw PBMC68k data
2. Filter low-quality genes (while protecting critical markers)
3. Normalize and log-transform
4. Save unsmoothed data for classification
5. Compute RNA velocity
6. Extract velocity and expression features
7. Normalize and boost features
8. Save processed data

The processed data includes both regular gene expression and velocity features.
At training time, you can choose which features to use via interactive prompts.

Usage:
    python src/prep_data.py
    python src/prep_data.py --plot  # Generate plots
"""
import scvelo as scv
import scanpy as sc
import numpy as np
import os
import argparse


# ============================================================================
# HELPER FUNCTIONS: Gene Filtering
# ============================================================================

def get_protected_genes():
    """
    Return lists of genes that should never be filtered.
    
    Returns:
        dict: Dictionary with keys 's_phase', 'g2m_phase', 'markers' containing gene lists
    """
    # S-phase cell cycle genes to ALWAYS keep
    s_genes = [
        'MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6', 
        'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'HELLS', 'RFC2', 'RPA2', 'NASP', 
        'RAD51AP1', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 
        'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 
        'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8'
    ]

    # G2M-phase cell cycle genes to ALWAYS keep
    g2m_genes = [
        'HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 
        'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'SMC4', 'CCNB2', 
        'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 
        'KIF20B', 'HJURP', 'CDCA3', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 
        'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1', 
        'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA'
    ]

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
    
    return {
        's_phase': s_genes,
        'g2m_phase': g2m_genes,
        'markers': cell_surface_markers
    }


def filter_low_quality_genes(adata, min_cells=10, remove_mt=True, remove_ribo=True, 
                              remove_pseudogenes=True, remove_olfactory=True):
    """
    Remove low-quality genes while protecting critical markers.
    
    Args:
        adata: AnnData object
        min_cells: Minimum number of cells expressing a gene
        remove_mt: Remove mitochondrial genes (MT-*)
        remove_ribo: Remove ribosomal genes (RPS*, RPL*)
        remove_pseudogenes: Remove pseudogenes (RP* but not RPS/RPL)
        remove_olfactory: Remove olfactory receptors (OR*)
    
    Returns:
        tuple: (filtered_adata, protected_genes_dict)
    """
    print("\nRemoving low-quality genes...")
    
    protected = get_protected_genes()
    all_protected = protected['s_phase'] + protected['g2m_phase'] + protected['markers']
    protected_upper = set([g.upper() for g in all_protected])
    
    print(f"   Protected gene categories:")
    print(f"     - S-phase cell cycle: {len(protected['s_phase'])} genes")
    print(f"     - G2M-phase cell cycle: {len(protected['g2m_phase'])} genes")
    print(f"     - Cell surface markers (Zheng 2017): {len(protected['markers'])} genes")
    
    # Identify genes to remove
    gene_names_upper = adata.var_names.str.upper()
    is_protected = gene_names_upper.isin(protected_upper)
    
    genes_to_remove = np.zeros(len(gene_names_upper), dtype=bool)
    
    if remove_pseudogenes:
        pseudogenes = gene_names_upper.str.match(r'^RP[^SL]')  # RP but not RPS or RPL
        genes_to_remove |= pseudogenes
        print(f"   Pseudogenes (RP*): {pseudogenes.sum()}")
    
    if remove_olfactory:
        olfactory = gene_names_upper.str.startswith('OR')
        genes_to_remove |= olfactory
        print(f"   Olfactory receptors (OR*): {olfactory.sum()}")
    
    if remove_mt:
        mitochondrial = gene_names_upper.str.startswith('MT-')
        genes_to_remove |= mitochondrial
        print(f"   Mitochondrial genes (MT-*): {mitochondrial.sum()}")
    
    if remove_ribo:
        ribosomal = gene_names_upper.str.startswith(('RPS', 'RPL'))
        genes_to_remove |= ribosomal
        print(f"   Ribosomal genes (RPS*, RPL*): {ribosomal.sum()}")
    
    # NEVER remove protected genes
    genes_to_remove = genes_to_remove & ~is_protected
    
    print(f"   Protected genes found: {is_protected.sum()}")
    print(f"   Total genes to remove: {genes_to_remove.sum()}")
    
    adata_filtered = adata[:, ~genes_to_remove].copy()
    print(f"   ✓ After filtering: {adata_filtered.shape[0]} cells x {adata_filtered.shape[1]} genes")
    
    return adata_filtered, protected


# ============================================================================
# HELPER FUNCTIONS: Velocity Computation
# ============================================================================

def compute_velocity_features(adata, n_pcs=30, n_neighbors=30):
    """
    Compute RNA velocity on preprocessed data.
    
    This function:
    1. Saves unsmoothed log-transformed data
    2. Computes PCA, neighbors, and moments (which smooths data)
    3. Estimates RNA velocity
    4. Computes velocity graph
    5. Computes UMAP
    
    Args:
        adata: AnnData object (must be normalized and log-transformed)
        n_pcs: Number of principal components for neighbor computation
        n_neighbors: Number of neighbors for smoothing
    
    Returns:
        AnnData object with velocity computed
    """
    print("\n   Computing PCA, neighbors, and moments...")
    
    # CRITICAL: Save unsmoothed log-transformed data for classification
    # The moments() function will smooth adata.X across neighbors, which helps velocity
    # estimation but hurts classification by blurring cell type boundaries
    adata.layers['unsmoothed_log'] = adata.X.copy()
    print(f"   ✓ Saved unsmoothed log-transformed data to layer 'unsmoothed_log' for classification")
    print(f"      (adata.X will be smoothed by moments, but unsmoothed version preserved)")
    
    # Compute PCA
    sc.tl.pca(adata, n_comps=50)
    print(f"   ✓ PCA computed (50 components)")

    # Compute neighbors in PCA space
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    print(f"   ✓ Neighbors computed ({n_pcs} PCs, {n_neighbors} neighbors)")

    # Compute moments using pre-computed neighbors
    # This smooths adata.X across neighbors (good for velocity, preserved unsmoothed for classification)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)  # Use pre-computed
    print(f"   ✓ Moments computed (adata.X is now smoothed for velocity estimation)")
    print(f"   ✓ Unsmoothed data available in layer 'unsmoothed_log' for classification")
    
    # Estimate RNA velocity
    print("\nEstimating RNA velocity...")
    scv.tl.velocity(adata, mode='stochastic')
    print(f"   ✓ Velocity shape: {adata.layers['velocity'].shape}")

    # Velocity Graph
    scv.tl.velocity_graph(adata, n_jobs=1)
    print("   ✓ Velocity graph computed")

    # Compute UMAP (needed for velocity embedding plots)
    print("\nComputing UMAP...")
    sc.tl.umap(adata)
    print("   ✓ UMAP computed")
    
    return adata


# ============================================================================
# HELPER FUNCTIONS: Feature Extraction
# ============================================================================

def extract_velocity_features(adata, protected_genes):
    """
    Extract 9 velocity-based features for classification.
    
    Features:
    1. velocity_pseudotime: Normalized velocity magnitude
    2. latent_time: Velocity direction consistency proxy
    3. velocity_confidence: Certainty of velocity estimate
    4. velocity_magnitude: Raw velocity magnitude
    5. S_score: S-phase cell cycle score
    6. G2M_score: G2M-phase cell cycle score
    7. mean_expression: Mean gene expression per cell
    8. expression_variance: Expression variance per cell
    9. n_genes_expressed: Number of genes expressed per cell
    
    Args:
        adata: AnnData object with velocity computed
        protected_genes: Dictionary with 's_phase', 'g2m_phase' keys
    
    Returns:
        AnnData object with features added to .obs
    """
    print("\nExtracting velocity features...")
    
    # Get velocity matrix for feature computation
    velocity_matrix = adata.layers['velocity']
    if hasattr(velocity_matrix, 'toarray'):
        velocity_matrix = velocity_matrix.toarray()
    
    # Compute velocity magnitude (L2 norm)
    velocity_magnitude = np.sqrt((velocity_matrix ** 2).sum(axis=1))
    
    # 1. Velocity pseudotime - normalized velocity magnitude
    adata.obs['velocity_pseudotime'] = (
        (velocity_magnitude - velocity_magnitude.min()) / 
        (velocity_magnitude.max() - velocity_magnitude.min())
    )
    print(f"   ✓ Added velocity_pseudotime (mean: {adata.obs['velocity_pseudotime'].mean():.4f})")

    # 2. Latent time - use velocity direction consistency as proxy
    velocity_direction = np.sign(velocity_matrix).mean(axis=1)
    
    # Ensure it's a flat 1D array
    if hasattr(velocity_direction, 'A1'):
        velocity_direction = velocity_direction.A1
    else:
        velocity_direction = np.asarray(velocity_direction).flatten()
    
    # Normalize to 0-1 range
    vd_min = velocity_direction.min()
    vd_max = velocity_direction.max()
    vd_range = vd_max - vd_min
    
    if vd_range < 1e-10:
        print(f"   ⚠️  All velocity directions identical, setting latent_time to 0.5")
        adata.obs['latent_time'] = np.full(len(velocity_direction), 0.5)
    else:
        adata.obs['latent_time'] = (velocity_direction - vd_min) / vd_range
    
    print(f"   ✓ Added latent_time proxy (mean: {adata.obs['latent_time'].mean():.4f})")

    # 3. Velocity confidence (certainty of velocity estimate)
    scv.tl.velocity_confidence(adata)
    print(f"   ✓ Added velocity_confidence (mean: {adata.obs['velocity_confidence'].mean():.4f})")

    # 4. Velocity magnitude (raw magnitude values)
    adata.obs['velocity_magnitude'] = velocity_magnitude
    print(f"   ✓ Added velocity_magnitude (mean: {velocity_magnitude.mean():.4f})")

    # 5-6. Cell cycle phase scores
    adata = compute_cell_cycle_scores(adata, protected_genes)
    
    # 7-9. Expression statistics
    adata = compute_expression_statistics(adata)
    
    return adata


def compute_cell_cycle_scores(adata, protected_genes):
    """
    Compute cell cycle scores using protected gene lists.
    
    Args:
        adata: AnnData object
        protected_genes: Dictionary with 's_phase', 'g2m_phase' keys
    
    Returns:
        AnnData object with S_score and G2M_score added to .obs
    """
    print("\nComputing cell cycle scores...")
    
    s_genes = protected_genes['s_phase']
    g2m_genes = protected_genes['g2m_phase']
    
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
            raise ValueError(f"Not enough cell cycle genes")
            
    except (ValueError, Exception) as e:
        print(f"   ⚠ Warning: {e}")
        print(f"   Using expression statistics as proxy for cell cycle activity")
        
        if hasattr(adata.X, 'toarray'):
            X_dense = adata.X.toarray()
        else:
            X_dense = adata.X
        
        # Use mean and variance as proxies
        mean_expr = X_dense.mean(axis=1)
        expr_var = X_dense.var(axis=1)
        
        adata.obs['S_score'] = (mean_expr - mean_expr.min()) / (mean_expr.max() - mean_expr.min())
        adata.obs['G2M_score'] = (expr_var - expr_var.min()) / (expr_var.max() - expr_var.min())
        
        print(f"   ✓ Added S_score proxy (mean: {adata.obs['S_score'].mean():.4f})")
        print(f"   ✓ Added G2M_score proxy (mean: {adata.obs['G2M_score'].mean():.4f})")
    
    return adata


def compute_expression_statistics(adata):
    """
    Compute per-cell expression statistics.
    
    Args:
        adata: AnnData object
    
    Returns:
        AnnData object with mean_expression, expression_variance, n_genes_expressed added to .obs
    """
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
    
    print(f"   ✓ Added mean_expression (mean: {mean_expr.mean():.4f})")
    print(f"   ✓ Added expression_variance (mean: {expr_var.mean():.4f})")
    print(f"   ✓ Added n_genes_expressed (mean: {n_genes.mean():.1f})")
    
    return adata


# ============================================================================
# HELPER FUNCTIONS: Feature Normalization
# ============================================================================

def normalize_and_boost_features(adata, boost_factor=1.5):
    """
    Normalize all velocity features to 0-1 range, then boost by a factor.
    
    This helps balance velocity features (which would be 0-1) with gene expression
    (which is 0-4.38 after log normalization). Boosting to 0-1.5 gives velocity
    features more influence in the classifier.
    
    Args:
        adata: AnnData object with features in .obs
        boost_factor: Multiplier after 0-1 normalization (default 1.5x)
    
    Returns:
        AnnData object with normalized and boosted features
    """
    print("\nNormalizing and boosting velocity features...")
    
    velocity_features = [
        'velocity_pseudotime', 'latent_time', 'velocity_confidence', 
        'velocity_magnitude', 'S_score', 'G2M_score', 
        'mean_expression', 'expression_variance', 'n_genes_expressed'
    ]
    
    print(f"   Boost factor: {boost_factor}x (to balance with gene expression 0-4.38 range)")
    
    for feat in velocity_features:
        if feat not in adata.obs.columns:
            continue
            
        values = adata.obs[feat].values
        
        # Check for NaNs before normalization
        nan_count = np.isnan(values).sum()
        if nan_count > 0:
            print(f"   ❌ ERROR: {feat} contains {nan_count} NaNs! Replacing with 0...")
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
        
        # Check for NaNs after normalization
        nan_count_after = np.isnan(normalized).sum()
        if nan_count_after > 0:
            print(f"   ❌ ERROR: {feat} produced {nan_count_after} NaNs after normalization!")
            normalized = np.nan_to_num(normalized, nan=0.0)
        
        # BOOST by factor to increase influence relative to genes
        boosted = normalized * boost_factor
        
        adata.obs[feat] = boosted
        print(f"   ✓ Normalized & boosted {feat}: range [{boosted.min():.4f}, {boosted.max():.4f}], mean={boosted.mean():.4f}")
    
    return adata


def check_data_quality(adata, verbose=True):
    """
    Check data quality and print diagnostics.
    
    Args:
        adata: AnnData object
        verbose: Print detailed statistics
    
    Returns:
        dict: Dictionary with data quality metrics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"DATA QUALITY DIAGNOSTICS")
        print(f"{'='*60}")
    
    # Convert to dense if needed
    if hasattr(adata.X, 'toarray'):
        X_check = adata.X.toarray()
    else:
        X_check = np.array(adata.X)
    
    # Count cells expressing each gene
    cells_per_gene = (X_check > 0).sum(axis=0)
    if len(cells_per_gene.shape) > 1:
        cells_per_gene = np.asarray(cells_per_gene).flatten()
    
    metrics = {
        'n_cells': adata.shape[0],
        'n_genes': adata.shape[1],
        'data_min': X_check.min(),
        'data_max': X_check.max(),
        'data_mean': X_check.mean(),
        'sparsity': (X_check == 0).sum() / X_check.size,
        'genes_expr_1cell': (cells_per_gene >= 1).sum(),
        'genes_expr_10cells': (cells_per_gene >= 10).sum(),
        'genes_expr_100cells': (cells_per_gene >= 100).sum(),
    }
    
    if verbose:
        print(f"Shape: {metrics['n_cells']} cells × {metrics['n_genes']} genes")
        print(f"\nExpression range:")
        print(f"  Min: {metrics['data_min']:.4f}")
        print(f"  Max: {metrics['data_max']:.4f}")
        print(f"  Mean: {metrics['data_mean']:.4f}")
        
        print(f"\nGene expression:")
        print(f"  Genes in ≥1 cells: {metrics['genes_expr_1cell']}")
        print(f"  Genes in ≥10 cells: {metrics['genes_expr_10cells']}")
        print(f"  Genes in ≥100 cells: {metrics['genes_expr_100cells']}")
        
        print(f"\nSparsity: {metrics['sparsity']*100:.1f}% zeros")
        print(f"{'='*60}\n")
    
    return metrics


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

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
