"""Extract velocity and expression features for classification."""
import numpy as np
import scvelo as scv


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
