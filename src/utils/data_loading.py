"""Data loading and validation utilities."""
import numpy as np
import anndata


def load_and_validate_data(file_path):
    """
    Load AnnData file and validate its contents.
    
    Args:
        file_path: Path to .h5ad file
    
    Returns:
        AnnData object
    
    Raises:
        ValueError: If data is invalid or missing required layers
    """
    print(f"Loading dataset from {file_path}...")
    adata = anndata.io.read_h5ad(file_path)
    
    print(f"  Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"  Layers: {list(adata.layers.keys())}")
    
    # Check for required layers
    if 'unsmoothed_log' not in adata.layers:
        print(f"  ⚠ Warning: 'unsmoothed_log' layer not found")
        print(f"     Using adata.X instead (may be smoothed, hurting classification)")
    else:
        print(f"  ✓ Found 'unsmoothed_log' layer for classification")
    
    # Check for velocity features
    velocity_features = ['velocity_pseudotime', 'latent_time', 'velocity_confidence', 
                         'velocity_magnitude', 'S_score', 'G2M_score']
    found_features = [f for f in velocity_features if f in adata.obs.columns]
    print(f"  Velocity features: {len(found_features)}/{len(velocity_features)} found")
    
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
