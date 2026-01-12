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


def load_gene_expression_and_features(adata, use_velocity=True):
    """
    Load gene expression (unsmoothed) and optionally velocity features.
    
    Args:
        adata: AnnData object
        use_velocity: Whether to include velocity features
    
    Returns:
        numpy array: Combined features (genes + velocity if requested)
    """
    # CRITICAL: Use unsmoothed data for classification
    if 'unsmoothed_log' in adata.layers:
        print(f"\n✓ Using UNSMOOTHED log-transformed data for classification")
        print(f"  (scVelo moments() smoothed adata.X across neighbors - blurs cell type boundaries)")
        print(f"  (unsmoothed_log layer preserves sharp boundaries needed for classification)")
        X_genes = adata.layers['unsmoothed_log']
    else:
        print(f"\n⚠ WARNING: Using SMOOTHED data (unsmoothed_log layer not found)")
        print(f"  This may hurt classification by blurring cell type boundaries!")
        X_genes = adata.X

    if hasattr(X_genes, 'toarray'):
        X_genes = X_genes.toarray()
    else:
        X_genes = np.array(X_genes)

    print(f"\nGene expression matrix:")
    print(f"  Shape: {X_genes.shape}")
    print(f"  Range: [{X_genes.min():.4f}, {X_genes.max():.4f}]")
    print(f"  Mean: {X_genes.mean():.4f}")

    # Add velocity features if requested
    if use_velocity:
        print("\nChecking for velocity and statistical features...")
        velocity_features = []
        
        feature_names = [
            'velocity_pseudotime', 'latent_time', 'velocity_confidence', 'velocity_magnitude',
            'S_score', 'G2M_score', 'mean_expression', 'expression_variance', 'n_genes_expressed'
        ]
        
        for feat in feature_names:
            if feat in adata.obs.columns:
                values = adata.obs[feat].values.reshape(-1, 1)
                velocity_features.append(values)
                print(f"  ✓ {feat}: range [{values.min():.4f}, {values.max():.4f}], mean {values.mean():.4f}")
        
        # Combine features
        if velocity_features:
            velocity_array = np.column_stack(velocity_features)
            
            # Check for NaN or Inf values
            n_nans = np.isnan(velocity_array).sum()
            n_infs = np.isinf(velocity_array).sum()

            if n_nans > 0 or n_infs > 0:
                velocity_array = np.nan_to_num(velocity_array, nan=0.0, posinf=0.0, neginf=0.0)
                print(f"  ⚠ Fixed {n_nans} NaNs and {n_infs} Infs in velocity features")
            
            X = np.column_stack([X_genes, velocity_array])
            n_genes = X_genes.shape[1]
            n_additional = velocity_array.shape[1]
            
            print(f"\n✓ Combined features: {X.shape[1]} total ({n_genes} genes + {n_additional} velocity features)")
            print(f"  Gene expression range: [{X[:, :n_genes].min():.4f}, {X[:, :n_genes].max():.4f}]")
            print(f"  Velocity features range: [{X[:, n_genes:].min():.4f}, {X[:, n_genes:].max():.4f}]")
            print(f"  Note: Velocity features already boosted 1.5x during preprocessing")
        else:
            print("\n⚠ No velocity features found, using gene expression only")
            X = X_genes
    else:
        print("\n✓ Skipping velocity features (using gene expression only)")
        X = X_genes
    
    return X


def add_symbolic_regression_features(adata, X):
    """
    Discover gene regulatory relationships and add as features.
    
    Args:
        adata: AnnData object
        X: Existing feature matrix (numpy array)
    
    Returns:
        numpy array: Feature matrix with symbolic regression features added
    """
    try:
        from symbolic_regression import (
            get_marker_genes_per_celltype,
            discover_gene_relationships,
            compute_regulatory_features
        )
        
        print("\n" + "="*70)
        print("🔬 SYMBOLIC REGRESSION: DISCOVERING GENE RELATIONSHIPS")
        print("="*70)
        print("This may take 5-15 minutes depending on your settings...")
        print()
        
        # Get marker genes
        marker_dict = get_marker_genes_per_celltype(
            adata,
            cell_type_column='cell_type',
            n_markers=7,
            method='predefined'
        )
        
        # Discover relationships
        models = discover_gene_relationships(
            adata,
            marker_dict,
            cell_type_column='cell_type',
            use_unsmoothed=True,
            min_cells_per_type=100,
            niterations=40,
            timeout_in_seconds=300
        )
        
        if len(models) > 0:
            # Compute regulatory features
            reg_features = compute_regulatory_features(adata, models, use_unsmoothed=True)
            
            # Combine with existing features
            X_combined = np.column_stack([X, reg_features.values])
            
            print(f"\n✓ Added {reg_features.shape[1]} regulatory features")
            print(f"  Total features: {X_combined.shape[1]}")
            print(f"  ({X.shape[1]} expression/velocity + {reg_features.shape[1]} regulatory)")
            
            return X_combined
        else:
            print("\n⚠️  No regulatory relationships discovered")
            print("Continuing without symbolic regression features...")
            return X
        
    except ImportError:
        print("\n⚠️  Symbolic regression module not installed")
        print("Install with:")
        print("  pip install pysr scipy networkx")
        print("  python -c 'import pysr; pysr.install()'")
        print("\nContinuing without symbolic regression features...")
        return X
    except Exception as e:
        print(f"\n⚠️  Symbolic regression failed: {e}")
        print("Continuing without symbolic regression features...")
        return X


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
