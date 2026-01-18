"""Automatic cell type hierarchy detection using expression profile clustering."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
import warnings


def compute_celltype_profiles(adata, cell_type_column: str = 'cell_type',
                              layer: str = None, use_hvg: bool = True,
                              n_hvg: int = 2000) -> Tuple[pd.DataFrame, List[str]]:
    """Compute mean expression profile for each cell type."""
    if layer and layer in adata.layers:
        X = adata.layers[layer]
    else:
        X = adata.X

    if hasattr(X, 'toarray'):
        X = X.toarray()

    gene_names = list(adata.var_names)

    if use_hvg and 'highly_variable' in adata.var.columns:
        hvg_mask = adata.var['highly_variable'].values
        X = X[:, hvg_mask]
        gene_names = [g for g, hv in zip(gene_names, hvg_mask) if hv]
    elif use_hvg:
        gene_var = np.var(X, axis=0)
        top_idx = np.argsort(gene_var)[-n_hvg:]
        X = X[:, top_idx]
        gene_names = [gene_names[i] for i in top_idx]

    cell_types = adata.obs[cell_type_column].values
    unique_types = sorted(set(cell_types))

    profiles = {}
    for ct in unique_types:
        mask = cell_types == ct
        profiles[ct] = np.mean(X[mask], axis=0)

    profiles_df = pd.DataFrame(profiles, index=gene_names).T
    return profiles_df, gene_names


def compute_similarity_matrix(profiles: pd.DataFrame, method: str = 'correlation') -> pd.DataFrame:
    """Compute pairwise similarity between cell type profiles."""
    cell_types = profiles.index.tolist()
    X = profiles.values

    if method == 'correlation':
        sim_matrix = np.corrcoef(X)
    elif method == 'cosine':
        sim_matrix = cosine_similarity(X)
    elif method == 'euclidean':
        dist_matrix = squareform(pdist(X, 'euclidean'))
        sim_matrix = 1 / (1 + dist_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")

    return pd.DataFrame(sim_matrix, index=cell_types, columns=cell_types)


def detect_hierarchy(adata, cell_type_column: str = 'cell_type', n_groups: int = None,
                     similarity_threshold: float = 0.5, method: str = 'correlation',
                     linkage_method: str = 'average', min_group_size: int = 1,
                     verbose: bool = True) -> Dict[str, List[str]]:
    """Automatically detect cell type hierarchy using clustering."""
    if verbose:
        print("\n" + "="*60)
        print("AUTO-DETECTING CELL TYPE HIERARCHY")
        print("="*60)

    cell_types = sorted(adata.obs[cell_type_column].unique())
    n_types = len(cell_types)

    if verbose:
        print(f"\nFound {n_types} cell types:")
        for ct in cell_types:
            count = (adata.obs[cell_type_column] == ct).sum()
            print(f"  - {ct}: {count} cells")

    if n_types <= 2:
        if verbose:
            print(f"\nOnly {n_types} cell types found, no hierarchy needed")
        return {'All_cells': cell_types}

    if verbose:
        print(f"\nComputing expression profiles...")

    profiles, genes = compute_celltype_profiles(adata, cell_type_column=cell_type_column, use_hvg=True)

    if verbose:
        print(f"  Using {len(genes)} genes")
        print(f"\nComputing {method} similarity...")

    sim_matrix = compute_similarity_matrix(profiles, method=method)

    sim_values = sim_matrix.values
    sim_values = np.clip(sim_values, -1, 1)
    dist_matrix = 1 - sim_values
    np.fill_diagonal(dist_matrix, 0)

    if verbose:
        print(f"\nPerforming hierarchical clustering (method={linkage_method})...")

    condensed_dist = squareform(dist_matrix, checks=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = linkage(condensed_dist, method=linkage_method)

    if n_groups is None:
        target_groups = max(3, min(int(np.sqrt(n_types) + 1), n_types - 1))
        heights = Z[:, 2]

        if len(heights) > 2:
            height_diffs = np.diff(heights)
            mean_diff = np.mean(height_diffs)
            std_diff = np.std(height_diffs)
            significant_jumps = np.where(height_diffs > mean_diff + std_diff)[0]

            if len(significant_jumps) > 0:
                best_distance = float('inf')
                for jump_idx in significant_jumps:
                    resulting_groups = n_types - jump_idx - 1
                    distance = abs(resulting_groups - target_groups)
                    if distance < best_distance:
                        best_distance = distance
                        n_groups = resulting_groups
                n_groups = max(3, min(n_groups, n_types - 1))
            else:
                n_groups = target_groups
        else:
            n_groups = target_groups

        if verbose:
            print(f"  Auto-selected {n_groups} groups")

    clusters = fcluster(Z, n_groups, criterion='maxclust')

    hierarchy = {}
    cluster_to_types = {}

    for ct, cluster_id in zip(cell_types, clusters):
        if cluster_id not in cluster_to_types:
            cluster_to_types[cluster_id] = []
        cluster_to_types[cluster_id].append(ct)

    for cluster_id, types in sorted(cluster_to_types.items()):
        group_name = _generate_group_name(types)
        hierarchy[group_name] = types

    if verbose:
        print(f"\n{'─'*60}")
        print("DETECTED HIERARCHY:")
        print(f"{'─'*60}")
        for group_name, types in hierarchy.items():
            print(f"\n  {group_name}:")
            for t in types:
                count = (adata.obs[cell_type_column] == t).sum()
                print(f"    - {t} ({count} cells)")

        print(f"\n{'─'*60}")
        print("WITHIN-GROUP SIMILARITIES:")
        print(f"{'─'*60}")
        for group_name, types in hierarchy.items():
            if len(types) > 1:
                group_sim = sim_matrix.loc[types, types].values
                triu_idx = np.triu_indices(len(types), k=1)
                mean_sim = np.mean(group_sim[triu_idx])
                min_sim = np.min(group_sim[triu_idx])
                print(f"  {group_name}: mean={mean_sim:.3f}, min={min_sim:.3f}")
            else:
                print(f"  {group_name}: single cell type")

    return hierarchy


def _generate_group_name(cell_types: List[str]) -> str:
    """Generate a meaningful name for a group of cell types."""
    if len(cell_types) == 1:
        return cell_types[0]

    common_words = {
        'T': ['T cell', 'T cells', 'CD4', 'CD8', 'T Reg', 'Helper', 'Cytotoxic'],
        'B': ['B cell', 'B cells', 'CD19', 'CD20'],
        'NK': ['NK', 'Natural Killer'],
        'Monocyte': ['Monocyte', 'CD14', 'CD16'],
        'Dendritic': ['Dendritic', 'DC', 'pDC', 'mDC'],
        'Myeloid': ['Myeloid', 'Monocyte', 'Macrophage', 'Granulocyte'],
        'Stem': ['Stem', 'Progenitor', 'CD34', 'HSC'],
    }

    for group_name, patterns in common_words.items():
        matches = 0
        for ct in cell_types:
            ct_lower = ct.lower()
            if any(p.lower() in ct_lower for p in patterns):
                matches += 1

        if matches == len(cell_types):
            return f"{group_name}_cells"
        elif matches >= len(cell_types) * 0.5:
            return f"{group_name}_related"

    if len(cell_types) >= 2:
        prefix = cell_types[0]
        for ct in cell_types[1:]:
            while not ct.startswith(prefix) and len(prefix) > 0:
                prefix = prefix[:-1]
        if len(prefix) >= 3:
            return f"{prefix.strip()}_group"

    return f"Group_{hash(tuple(sorted(cell_types))) % 1000}"


def visualize_hierarchy(adata, hierarchy: Dict[str, List[str]] = None,
                        cell_type_column: str = 'cell_type', save_path: str = None,
                        figsize: Tuple[int, int] = (12, 8)):
    """Visualize the cell type hierarchy as a dendrogram."""
    import matplotlib.pyplot as plt

    profiles, _ = compute_celltype_profiles(adata, cell_type_column=cell_type_column)
    sim_matrix = compute_similarity_matrix(profiles, method='correlation')

    dist_matrix = 1 - sim_matrix.values
    np.fill_diagonal(dist_matrix, 0)
    condensed_dist = squareform(dist_matrix, checks=False)
    Z = linkage(condensed_dist, method='average')

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1 = axes[0]
    dendrogram(Z, labels=profiles.index.tolist(), orientation='left', ax=ax1, leaf_font_size=10)
    ax1.set_title('Cell Type Hierarchy (Dendrogram)')
    ax1.set_xlabel('Distance (1 - correlation)')

    ax2 = axes[1]
    im = ax2.imshow(sim_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_xticks(range(len(sim_matrix)))
    ax2.set_yticks(range(len(sim_matrix)))
    ax2.set_xticklabels(sim_matrix.columns, rotation=45, ha='right', fontsize=8)
    ax2.set_yticklabels(sim_matrix.index, fontsize=8)
    ax2.set_title('Cell Type Similarity Matrix')
    plt.colorbar(im, ax=ax2, label='Correlation')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved hierarchy visualization to {save_path}")

    plt.show()


def get_hierarchy_for_classifier(adata, cell_type_column: str = 'cell_type',
                                  n_groups: int = None, similarity_threshold: float = 0.7,
                                  verbose: bool = True) -> Dict[str, List[str]]:
    """Convenience function to get hierarchy ready for the classifier."""
    return detect_hierarchy(
        adata,
        cell_type_column=cell_type_column,
        n_groups=n_groups,
        similarity_threshold=similarity_threshold,
        verbose=verbose
    )
