"""
Feature engineering using discovered symbolic relationships.

This module computes "regulatory scores" for each cell based on how well
its gene expression matches the discovered equations. These scores become
new features for the classifier.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pysr import PySRRegressor


def compute_regulatory_features(
    adata,
    models: Dict[str, Dict[str, PySRRegressor]],
    use_unsmoothed: bool = True
) -> pd.DataFrame:
    """
    Compute regulatory relationship features for all cells.
    
    For each discovered equation, this function:
    1. Predicts the target gene from predictor genes
    2. Compares prediction to actual expression
    3. Computes a "fit score" (how well the cell matches the relationship)
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    models : Dict[str, Dict[str, PySRRegressor]]
        Discovered models from discover_gene_relationships()
    use_unsmoothed : bool
        Use unsmoothed data (should match what was used for discovery)
        
    Returns
    -------
    features_df : pd.DataFrame
        DataFrame with regulatory features for each cell
        Columns: '<celltype>_<gene>_regulatory_score'
        
    Examples
    --------
    >>> models = discover_gene_relationships(adata, markers)
    >>> reg_features = compute_regulatory_features(adata, models)
    >>> print(reg_features.columns)
    # ['T_cells_CD3D_regulatory_score', 'T_cells_CD3E_regulatory_score', ...]
    """
    
    print(f"\n{'='*60}")
    print(f"PHASE 2: COMPUTING REGULATORY FEATURES")
    print(f"{'='*60}\n")
    
    # Get expression data
    if use_unsmoothed and 'unsmoothed_log' in adata.layers:
        expression_data = adata.layers['unsmoothed_log']
    else:
        expression_data = adata.X
    
    if hasattr(expression_data, 'toarray'):
        expression_data = expression_data.toarray()
    
    expr_df = pd.DataFrame(
        expression_data,
        index=adata.obs_names,
        columns=adata.var_names
    )
    
    # Store all regulatory features
    regulatory_features = {}
    
    for cell_type, ct_models in models.items():
        print(f"\nCell type: {cell_type}")
        print(f"  Computing {len(ct_models)} regulatory scores...")
        
        for target_gene, model in ct_models.items():
            # Get predictor genes from model - try different attribute names for compatibility
            if hasattr(model, 'feature_names_in_'):
                predictor_genes = model.feature_names_in_
            elif hasattr(model, 'variable_names'):
                predictor_genes = model.variable_names
            else:
                print(f"    ⚠ Cannot find feature names for {target_gene}, skipping")
                continue
            
            # Get expression data for all cells
            X = expr_df[predictor_genes].values
            y_actual = expr_df[target_gene].values
            
            # Predict using discovered equation
            try:
                y_pred = model.predict(X)
                
                # Compute regulatory score as inverse of prediction error
                # Multiple scoring options:
                
                # Option 1: Negative absolute error (higher = better fit)
                abs_error = np.abs(y_pred - y_actual)
                score = -abs_error
                
                # Option 2: R² per cell (correlation between prediction and actual)
                # squared_error = (y_pred - y_actual) ** 2
                # score = -squared_error
                
                # Option 3: Probability-like score (scaled to 0-1)
                # max_error = abs_error.max()
                # score = 1 - (abs_error / (max_error + 1e-8))
                
                # Store feature
                feature_name = f"{cell_type.replace(' ', '_')}_{target_gene}_reg_score"
                regulatory_features[feature_name] = score
                
                print(f"    ✓ {target_gene}: mean_score={score.mean():.3f}, std={score.std():.3f}")
                
            except Exception as e:
                print(f"    ✗ Failed to compute score for {target_gene}: {e}")
                continue
    
    # Convert to DataFrame
    features_df = pd.DataFrame(regulatory_features, index=adata.obs_names)
    
    print(f"\n{'='*60}")
    print(f"FEATURE ENGINEERING COMPLETE")
    print(f"{'='*60}")
    print(f"Generated {features_df.shape[1]} regulatory features")
    print(f"Feature matrix shape: {features_df.shape}")
    
    return features_df


def create_augmented_feature_matrix(
    adata,
    regulatory_features: pd.DataFrame,
    use_unsmoothed: bool = True,
    include_velocity_features: bool = True,
    gene_subset: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create augmented feature matrix combining gene expression and regulatory features.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    regulatory_features : pd.DataFrame
        Regulatory features from compute_regulatory_features()
    use_unsmoothed : bool
        Use unsmoothed gene expression
    include_velocity_features : bool
        Include the 9 RNA velocity features
    gene_subset : Optional[List[str]]
        Use only these genes (e.g., highly variable genes)
        If None, uses all genes
        
    Returns
    -------
    augmented_df : pd.DataFrame
        Combined feature matrix with:
        - Gene expression features
        - Velocity features (if requested)
        - Regulatory features
        
    Examples
    --------
    >>> reg_features = compute_regulatory_features(adata, models)
    >>> X_augmented = create_augmented_feature_matrix(adata, reg_features)
    >>> print(X_augmented.shape)  # (n_cells, n_genes + 9 velocity + n_regulatory)
    """
    
    print(f"\n{'='*60}")
    print(f"CREATING AUGMENTED FEATURE MATRIX")
    print(f"{'='*60}\n")
    
    # 1. Gene expression features
    if use_unsmoothed and 'unsmoothed_log' in adata.layers:
        expression_data = adata.layers['unsmoothed_log']
        print("Using unsmoothed gene expression")
    else:
        expression_data = adata.X
        print("Using smoothed gene expression")
    
    if hasattr(expression_data, 'toarray'):
        expression_data = expression_data.toarray()
    
    # Select gene subset if specified
    if gene_subset is not None:
        gene_mask = adata.var_names.isin(gene_subset)
        expression_data = expression_data[:, gene_mask]
        gene_names = adata.var_names[gene_mask].tolist()
        print(f"Using {len(gene_names)} selected genes")
    else:
        gene_names = adata.var_names.tolist()
        print(f"Using all {len(gene_names)} genes")
    
    expr_df = pd.DataFrame(
        expression_data,
        index=adata.obs_names,
        columns=gene_names
    )
    
    # 2. Velocity features (optional)
    velocity_features = [
        'velocity_pseudotime', 'latent_time', 'velocity_confidence',
        'velocity_magnitude', 'S_score', 'G2M_score',
        'mean_expression', 'expression_variance', 'n_genes_expressed'
    ]
    
    if include_velocity_features:
        available_velocity = [f for f in velocity_features if f in adata.obs.columns]
        if len(available_velocity) > 0:
            velocity_df = adata.obs[available_velocity].copy()
            print(f"Including {len(available_velocity)} velocity features")
        else:
            velocity_df = pd.DataFrame(index=adata.obs_names)
            print("No velocity features found")
    else:
        velocity_df = pd.DataFrame(index=adata.obs_names)
        print("Skipping velocity features")
    
    # 3. Combine all features
    augmented_df = pd.concat([expr_df, velocity_df, regulatory_features], axis=1)
    
    print(f"\n{'='*60}")
    print(f"AUGMENTED MATRIX SUMMARY")
    print(f"{'='*60}")
    print(f"Total features: {augmented_df.shape[1]}")
    print(f"  Gene expression: {expr_df.shape[1]}")
    print(f"  Velocity features: {velocity_df.shape[1]}")
    print(f"  Regulatory features: {regulatory_features.shape[1]}")
    print(f"Total cells: {augmented_df.shape[0]}")
    
    return augmented_df


def normalize_regulatory_features(
    features_df: pd.DataFrame,
    method: str = 'zscore'
) -> pd.DataFrame:
    """
    Normalize regulatory features to have similar scale as gene expression.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Regulatory features
    method : str
        'zscore': standardize to mean=0, std=1
        'minmax': scale to [0, 1]
        'robust': use median and IQR (robust to outliers)
        
    Returns
    -------
    normalized_df : pd.DataFrame
        Normalized features
    """
    
    if method == 'zscore':
        # Z-score normalization
        normalized_df = (features_df - features_df.mean()) / (features_df.std() + 1e-8)
    
    elif method == 'minmax':
        # Min-max scaling to [0, 1]
        min_vals = features_df.min()
        max_vals = features_df.max()
        normalized_df = (features_df - min_vals) / (max_vals - min_vals + 1e-8)
    
    elif method == 'robust':
        # Robust scaling using median and IQR
        median = features_df.median()
        q75 = features_df.quantile(0.75)
        q25 = features_df.quantile(0.25)
        iqr = q75 - q25
        normalized_df = (features_df - median) / (iqr + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_df
