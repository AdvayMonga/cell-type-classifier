"""Data loading and validation utilities."""
import numpy as np
import anndata


def load_and_validate_data(file_path):
    """Load AnnData file and validate its contents."""
    print(f"Loading dataset from {file_path}...")
    adata = anndata.io.read_h5ad(file_path)

    print(f"  Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"  Layers: {list(adata.layers.keys())}")

    if 'unsmoothed_log' not in adata.layers:
        print(f"  Warning: 'unsmoothed_log' layer not found, using adata.X")
    else:
        print(f"  Found 'unsmoothed_log' layer for classification")

    velocity_features = ['velocity_pseudotime', 'latent_time', 'velocity_confidence',
                         'velocity_magnitude', 'S_score', 'G2M_score']
    found_features = [f for f in velocity_features if f in adata.obs.columns]
    print(f"  Velocity features: {len(found_features)}/{len(velocity_features)} found")

    return adata


def load_gene_expression_and_features(adata, use_velocity=True):
    """Load gene expression (unsmoothed) and optionally velocity features."""
    if 'unsmoothed_log' in adata.layers:
        print(f"\nUsing UNSMOOTHED log-transformed data for classification")
        X_genes = adata.layers['unsmoothed_log']
    else:
        print(f"\nWARNING: Using SMOOTHED data (unsmoothed_log layer not found)")
        X_genes = adata.X

    if hasattr(X_genes, 'toarray'):
        X_genes = X_genes.toarray()
    else:
        X_genes = np.array(X_genes)

    print(f"\nGene expression: {X_genes.shape}, range [{X_genes.min():.4f}, {X_genes.max():.4f}]")

    if use_velocity:
        print("\nChecking for velocity features...")
        velocity_features = []

        feature_names = [
            'velocity_pseudotime', 'latent_time', 'velocity_confidence', 'velocity_magnitude',
            'S_score', 'G2M_score', 'mean_expression', 'expression_variance', 'n_genes_expressed'
        ]

        for feat in feature_names:
            if feat in adata.obs.columns:
                values = adata.obs[feat].values.reshape(-1, 1)
                velocity_features.append(values)
                print(f"  {feat}: [{values.min():.4f}, {values.max():.4f}]")

        if velocity_features:
            velocity_array = np.column_stack(velocity_features)

            n_nans = np.isnan(velocity_array).sum()
            n_infs = np.isinf(velocity_array).sum()

            if n_nans > 0 or n_infs > 0:
                velocity_array = np.nan_to_num(velocity_array, nan=0.0, posinf=0.0, neginf=0.0)
                print(f"  Fixed {n_nans} NaNs and {n_infs} Infs")

            X = np.column_stack([X_genes, velocity_array])
            print(f"\nCombined: {X.shape[1]} features ({X_genes.shape[1]} genes + {velocity_array.shape[1]} velocity)")
        else:
            print("\nNo velocity features found")
            X = X_genes
    else:
        print("\nUsing gene expression only")
        X = X_genes

    return X


def add_symbolic_regression_features(adata, X):
    """Discover gene regulatory relationships and add as features."""
    try:
        from symbolic_regression import (
            get_marker_genes_per_celltype,
            discover_gene_relationships,
            compute_regulatory_features
        )

        print("\n" + "="*70)
        print("SYMBOLIC REGRESSION: DISCOVERING GENE RELATIONSHIPS")
        print("="*70)

        marker_dict = get_marker_genes_per_celltype(
            adata, cell_type_column='cell_type', n_markers=7, method='predefined'
        )

        models = discover_gene_relationships(
            adata, marker_dict, cell_type_column='cell_type',
            use_unsmoothed=True, min_cells_per_type=100,
            niterations=40, timeout_in_seconds=300
        )

        if len(models) > 0:
            reg_features = compute_regulatory_features(adata, models, use_unsmoothed=True)
            X_combined = np.column_stack([X, reg_features.values])
            print(f"\nAdded {reg_features.shape[1]} regulatory features, total: {X_combined.shape[1]}")
            return X_combined
        else:
            print("\nNo regulatory relationships discovered")
            return X

    except ImportError:
        print("\nSymbolic regression module not installed")
        return X
    except Exception as e:
        print(f"\nSymbolic regression failed: {e}")
        return X


def check_data_quality(adata, verbose=True):
    """Check data quality and print diagnostics."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"DATA QUALITY DIAGNOSTICS")
        print(f"{'='*60}")

    if hasattr(adata.X, 'toarray'):
        X_check = adata.X.toarray()
    else:
        X_check = np.array(adata.X)

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
        print(f"Shape: {metrics['n_cells']} cells x {metrics['n_genes']} genes")
        print(f"Range: [{metrics['data_min']:.4f}, {metrics['data_max']:.4f}], Mean: {metrics['data_mean']:.4f}")
        print(f"Genes in >=1/10/100 cells: {metrics['genes_expr_1cell']}/{metrics['genes_expr_10cells']}/{metrics['genes_expr_100cells']}")
        print(f"Sparsity: {metrics['sparsity']*100:.1f}% zeros")
        print(f"{'='*60}\n")

    return metrics
