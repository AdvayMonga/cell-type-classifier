"""Pathway scoring for biological feature engineering."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# Curated immune cell pathway gene sets
IMMUNE_PATHWAYS = {
    'T_cell_activation': [
        'CD3D', 'CD3E', 'CD3G', 'CD28', 'LCK', 'ZAP70', 'LAT', 'ITK'
    ],
    'T_cell_cytotoxicity': [
        'GZMA', 'GZMB', 'GZMK', 'PRF1', 'GNLY', 'NKG7', 'FASLG'
    ],
    'B_cell_activation': [
        'CD19', 'MS4A1', 'CD79A', 'CD79B', 'BLK', 'CD22', 'PAX5'
    ],
    'NK_cell_signature': [
        'NCAM1', 'NKG7', 'GNLY', 'KLRD1', 'KLRB1', 'KLRC1', 'NCR1'
    ],
    'monocyte_signature': [
        'CD14', 'LYZ', 'S100A8', 'S100A9', 'FCN1', 'VCAN', 'CD68'
    ],
    'dendritic_cell_signature': [
        'FCER1A', 'CD1C', 'CLEC10A', 'IL3RA', 'IRF4', 'IRF8'
    ],
    'cell_cycle_G1S': [
        'MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG'
    ],
    'cell_cycle_G2M': [
        'HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A'
    ],
    'interferon_response': [
        'ISG15', 'ISG20', 'IFI6', 'IFIT1', 'IFIT3', 'MX1', 'OAS1'
    ],
    'antigen_presentation': [
        'HLA-A', 'HLA-B', 'HLA-C', 'HLA-DRA', 'HLA-DRB1', 'B2M', 'TAP1'
    ],
    'inflammatory_response': [
        'IL1B', 'IL6', 'TNF', 'CXCL8', 'CCL2', 'CCL3', 'CCL4'
    ],
    'apoptosis': [
        'BAX', 'BCL2', 'CASP3', 'CASP8', 'CASP9', 'CYCS', 'APAF1'
    ],
}


def compute_pathway_scores(adata, pathways: Optional[Dict[str, List[str]]] = None,
                           use_unsmoothed: bool = True, method: str = 'mean') -> pd.DataFrame:
    """
    Compute pathway activity scores for each cell.

    Args:
        adata: AnnData object with gene expression
        pathways: Dict mapping pathway names to gene lists (uses IMMUNE_PATHWAYS if None)
        use_unsmoothed: Use unsmoothed_log layer if available
        method: Scoring method ('mean', 'sum', or 'median')

    Returns:
        DataFrame with pathway scores (cells x pathways)
    """
    if pathways is None:
        pathways = IMMUNE_PATHWAYS

    if use_unsmoothed and 'unsmoothed_log' in adata.layers:
        expr_data = adata.layers['unsmoothed_log']
    else:
        expr_data = adata.X

    if hasattr(expr_data, 'toarray'):
        expr_data = expr_data.toarray()

    gene_names = list(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    scores = {}
    for pathway_name, genes in pathways.items():
        present_genes = [g for g in genes if g in gene_to_idx]

        if len(present_genes) < 2:
            continue

        gene_indices = [gene_to_idx[g] for g in present_genes]
        pathway_expr = expr_data[:, gene_indices]

        if method == 'mean':
            score = np.mean(pathway_expr, axis=1)
        elif method == 'sum':
            score = np.sum(pathway_expr, axis=1)
        elif method == 'median':
            score = np.median(pathway_expr, axis=1)
        else:
            raise ValueError(f"Unknown method: {method}")

        scores[f'pathway_{pathway_name}'] = score

    scores_df = pd.DataFrame(scores, index=adata.obs_names)
    return scores_df


def compute_gene_module_scores(adata, use_unsmoothed: bool = True,
                                n_bins: int = 25) -> pd.DataFrame:
    """
    Compute gene module scores using binned control gene approach.

    Similar to Scanpy's score_genes but returns a DataFrame.
    Scores are computed relative to control genes with similar expression.

    Args:
        adata: AnnData object
        use_unsmoothed: Use unsmoothed_log layer if available
        n_bins: Number of expression bins for control gene selection

    Returns:
        DataFrame with module scores
    """
    if use_unsmoothed and 'unsmoothed_log' in adata.layers:
        expr_data = adata.layers['unsmoothed_log']
    else:
        expr_data = adata.X

    if hasattr(expr_data, 'toarray'):
        expr_data = expr_data.toarray()

    gene_names = list(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    mean_expr = np.mean(expr_data, axis=0)
    bins = pd.cut(mean_expr, bins=n_bins, labels=False)

    scores = {}
    for pathway_name, genes in IMMUNE_PATHWAYS.items():
        present_genes = [g for g in genes if g in gene_to_idx]
        if len(present_genes) < 2:
            continue

        gene_indices = [gene_to_idx[g] for g in present_genes]
        module_expr = np.mean(expr_data[:, gene_indices], axis=1)

        control_indices = []
        for idx in gene_indices:
            bin_val = bins[idx]
            same_bin = np.where(bins == bin_val)[0]
            same_bin = [i for i in same_bin if i not in gene_indices]
            if len(same_bin) > 0:
                np.random.seed(42)
                ctrl = np.random.choice(same_bin, min(len(same_bin), 50), replace=False)
                control_indices.extend(ctrl)

        if len(control_indices) > 0:
            control_indices = list(set(control_indices))
            control_expr = np.mean(expr_data[:, control_indices], axis=1)
            score = module_expr - control_expr
        else:
            score = module_expr

        scores[f'module_{pathway_name}'] = score

    return pd.DataFrame(scores, index=adata.obs_names)


def add_pathway_features(adata, X: np.ndarray, method: str = 'mean',
                          include_modules: bool = False) -> np.ndarray:
    """
    Add pathway scores as features to existing feature matrix.

    Args:
        adata: AnnData object
        X: Existing feature matrix (n_cells, n_features)
        method: Scoring method for pathways
        include_modules: Also include module scores (slower)

    Returns:
        Augmented feature matrix with pathway scores
    """
    print("\nComputing pathway scores...")

    pathway_df = compute_pathway_scores(adata, method=method)
    print(f"  Pathway features: {pathway_df.shape[1]}")

    if include_modules:
        module_df = compute_gene_module_scores(adata)
        print(f"  Module features: {module_df.shape[1]}")
        all_scores = pd.concat([pathway_df, module_df], axis=1)
    else:
        all_scores = pathway_df

    all_scores = all_scores.fillna(0)

    X_augmented = np.column_stack([X, all_scores.values])
    print(f"  Total features: {X_augmented.shape[1]} ({X.shape[1]} original + {all_scores.shape[1]} pathway)")

    return X_augmented


def get_pathway_names() -> List[str]:
    """Return list of available pathway names."""
    return list(IMMUNE_PATHWAYS.keys())


def get_pathway_genes(pathway_name: str) -> List[str]:
    """Return genes in a specific pathway."""
    return IMMUNE_PATHWAYS.get(pathway_name, [])
