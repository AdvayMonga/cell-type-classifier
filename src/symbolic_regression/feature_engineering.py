"""Feature engineering using discovered symbolic relationships."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pysr import PySRRegressor


def compute_regulatory_features(adata, models: Dict[str, Dict[str, PySRRegressor]],
                                 use_unsmoothed: bool = True) -> pd.DataFrame:
    """Compute regulatory relationship features for all cells."""

    print(f"\n{'='*60}")
    print(f"COMPUTING REGULATORY FEATURES")
    print(f"{'='*60}\n")

    if use_unsmoothed and 'unsmoothed_log' in adata.layers:
        expression_data = adata.layers['unsmoothed_log']
    else:
        expression_data = adata.X

    if hasattr(expression_data, 'toarray'):
        expression_data = expression_data.toarray()

    expr_df = pd.DataFrame(expression_data, index=adata.obs_names, columns=adata.var_names)
    regulatory_features = {}

    for cell_type, ct_models in models.items():
        print(f"\n{cell_type}: {len(ct_models)} scores...")

        for target_gene, model in ct_models.items():
            if hasattr(model, 'feature_names_in_'):
                predictor_genes = model.feature_names_in_
            elif hasattr(model, 'variable_names'):
                predictor_genes = model.variable_names
            else:
                print(f"    Cannot find feature names for {target_gene}")
                continue

            X = expr_df[predictor_genes].values
            y_actual = expr_df[target_gene].values

            try:
                y_pred = model.predict(X)
                abs_error = np.abs(y_pred - y_actual)
                score = -abs_error

                feature_name = f"{cell_type.replace(' ', '_')}_{target_gene}_reg_score"
                regulatory_features[feature_name] = score

                print(f"    {target_gene}: mean={score.mean():.3f}, std={score.std():.3f}")

            except Exception as e:
                print(f"    Failed for {target_gene}: {e}")

    features_df = pd.DataFrame(regulatory_features, index=adata.obs_names)

    print(f"\n{'='*60}")
    print(f"Generated {features_df.shape[1]} regulatory features")
    print(f"{'='*60}")

    return features_df


def create_augmented_feature_matrix(adata, regulatory_features: pd.DataFrame,
                                     use_unsmoothed: bool = True,
                                     include_velocity_features: bool = True,
                                     gene_subset: Optional[List[str]] = None) -> pd.DataFrame:
    """Create augmented feature matrix combining gene expression and regulatory features."""

    print(f"\n{'='*60}")
    print(f"CREATING AUGMENTED FEATURE MATRIX")
    print(f"{'='*60}\n")

    if use_unsmoothed and 'unsmoothed_log' in adata.layers:
        expression_data = adata.layers['unsmoothed_log']
        print("Using unsmoothed gene expression")
    else:
        expression_data = adata.X
        print("Using smoothed gene expression")

    if hasattr(expression_data, 'toarray'):
        expression_data = expression_data.toarray()

    if gene_subset is not None:
        gene_mask = adata.var_names.isin(gene_subset)
        expression_data = expression_data[:, gene_mask]
        gene_names = adata.var_names[gene_mask].tolist()
        print(f"Using {len(gene_names)} selected genes")
    else:
        gene_names = adata.var_names.tolist()
        print(f"Using all {len(gene_names)} genes")

    expr_df = pd.DataFrame(expression_data, index=adata.obs_names, columns=gene_names)

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

    augmented_df = pd.concat([expr_df, velocity_df, regulatory_features], axis=1)

    print(f"\nTotal: {augmented_df.shape[1]} features ({expr_df.shape[1]} genes + {velocity_df.shape[1]} velocity + {regulatory_features.shape[1]} regulatory)")

    return augmented_df


def normalize_regulatory_features(features_df: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
    """Normalize regulatory features."""

    if method == 'zscore':
        normalized_df = (features_df - features_df.mean()) / (features_df.std() + 1e-8)
    elif method == 'minmax':
        min_vals = features_df.min()
        max_vals = features_df.max()
        normalized_df = (features_df - min_vals) / (max_vals - min_vals + 1e-8)
    elif method == 'robust':
        median = features_df.median()
        q75 = features_df.quantile(0.75)
        q25 = features_df.quantile(0.25)
        iqr = q75 - q25
        normalized_df = (features_df - median) / (iqr + 1e-8)
    else:
        raise ValueError(f"Unknown method: {method}")

    return normalized_df
