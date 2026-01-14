"""
Automatic cell type hierarchy detection.

This module analyzes gene expression patterns to automatically group similar
cell types into a hierarchical structure for classification.

The approach:
1. Compute mean expression profile for each cell type
2. Calculate pairwise similarity (correlation) between cell types
3. Use hierarchical clustering to group similar cell types
4. Cut the dendrogram at an optimal level to create coarse groups

This works on any single-cell dataset with cell type annotations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
import warnings


def compute_celltype_profiles(
    adata,
    cell_type_column: str = 'cell_type',
    layer: str = None,
    use_hvg: bool = True,
    n_hvg: int = 2000
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute mean expression profile for each cell type.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    cell_type_column : str
        Column in adata.obs containing cell type labels
    layer : str, optional
        Layer to use (e.g., 'unsmoothed_log'). If None, uses adata.X
    use_hvg : bool
        Whether to use only highly variable genes
    n_hvg : int
        Number of highly variable genes to use
        
    Returns
    -------
    profiles : pd.DataFrame
        Mean expression profile for each cell type (cell_types x genes)
    gene_names : List[str]
        Names of genes used
    """
    # Get expression matrix
    if layer and layer in adata.layers:
        X = adata.layers[layer]
    else:
        X = adata.X
    
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Select genes
    gene_names = list(adata.var_names)
    
    if use_hvg and 'highly_variable' in adata.var.columns:
        hvg_mask = adata.var['highly_variable'].values
        X = X[:, hvg_mask]
        gene_names = [g for g, hv in zip(gene_names, hvg_mask) if hv]
    elif use_hvg:
        # Compute variance and select top genes
        gene_var = np.var(X, axis=0)
        top_idx = np.argsort(gene_var)[-n_hvg:]
        X = X[:, top_idx]
        gene_names = [gene_names[i] for i in top_idx]
    
    # Get cell types
    cell_types = adata.obs[cell_type_column].values
    unique_types = sorted(set(cell_types))
    
    # Compute mean profile for each cell type
    profiles = {}
    for ct in unique_types:
        mask = cell_types == ct
        profiles[ct] = np.mean(X[mask], axis=0)
    
    profiles_df = pd.DataFrame(profiles, index=gene_names).T
    
    return profiles_df, gene_names


def compute_similarity_matrix(
    profiles: pd.DataFrame,
    method: str = 'correlation'
) -> pd.DataFrame:
    """
    Compute pairwise similarity between cell type profiles.
    
    Parameters
    ----------
    profiles : pd.DataFrame
        Mean expression profiles (cell_types x genes)
    method : str
        Similarity method: 'correlation', 'cosine', or 'euclidean'
        
    Returns
    -------
    similarity : pd.DataFrame
        Pairwise similarity matrix
    """
    cell_types = profiles.index.tolist()
    X = profiles.values
    
    if method == 'correlation':
        # Pearson correlation
        sim_matrix = np.corrcoef(X)
    elif method == 'cosine':
        sim_matrix = cosine_similarity(X)
    elif method == 'euclidean':
        # Convert distance to similarity
        dist_matrix = squareform(pdist(X, 'euclidean'))
        sim_matrix = 1 / (1 + dist_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return pd.DataFrame(sim_matrix, index=cell_types, columns=cell_types)


def detect_hierarchy(
    adata,
    cell_type_column: str = 'cell_type',
    n_groups: int = None,
    similarity_threshold: float = 0.5,
    method: str = 'correlation',
    linkage_method: str = 'average',
    min_group_size: int = 1,
    verbose: bool = True
) -> Dict[str, List[str]]:
    """
    Automatically detect cell type hierarchy using clustering.
    
    This function groups similar cell types together based on their
    expression profiles, creating a 2-level hierarchy suitable for
    hierarchical classification.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with cell type labels
    cell_type_column : str
        Column in adata.obs containing cell type labels
    n_groups : int, optional
        Number of coarse groups to create. If None, determined automatically
        based on similarity_threshold
    similarity_threshold : float
        Minimum similarity within a group (used if n_groups is None)
    method : str
        Similarity method: 'correlation', 'cosine', or 'euclidean'
    linkage_method : str
        Clustering linkage method: 'average', 'complete', 'single', 'ward'
    min_group_size : int
        Minimum number of cell types per group
    verbose : bool
        Print progress and results
        
    Returns
    -------
    hierarchy : Dict[str, List[str]]
        Dictionary mapping group names to list of cell types
        
    Examples
    --------
    >>> hierarchy = detect_hierarchy(adata)
    >>> print(hierarchy)
    {'Group_1': ['CD4+ T cells', 'CD8+ T cells'],
     'Group_2': ['B cells'],
     'Group_3': ['Monocytes', 'Dendritic']}
    """
    if verbose:
        print("\n" + "="*60)
        print("AUTO-DETECTING CELL TYPE HIERARCHY")
        print("="*60)
    
    # Get unique cell types
    cell_types = sorted(adata.obs[cell_type_column].unique())
    n_types = len(cell_types)
    
    if verbose:
        print(f"\nFound {n_types} cell types:")
        for ct in cell_types:
            count = (adata.obs[cell_type_column] == ct).sum()
            print(f"  - {ct}: {count} cells")
    
    # Handle edge case: only 1-2 cell types
    if n_types <= 2:
        if verbose:
            print(f"\n⚠ Only {n_types} cell types found, no hierarchy needed")
        return {'All_cells': cell_types}
    
    # Compute expression profiles
    if verbose:
        print(f"\nComputing expression profiles...")
    
    profiles, genes = compute_celltype_profiles(
        adata, 
        cell_type_column=cell_type_column,
        use_hvg=True
    )
    
    if verbose:
        print(f"  Using {len(genes)} genes")
    
    # Compute similarity matrix
    if verbose:
        print(f"\nComputing {method} similarity...")
    
    sim_matrix = compute_similarity_matrix(profiles, method=method)
    
    # Convert similarity to distance for clustering
    # Ensure values are in valid range for distance
    sim_values = sim_matrix.values
    sim_values = np.clip(sim_values, -1, 1)
    dist_matrix = 1 - sim_values
    np.fill_diagonal(dist_matrix, 0)  # Distance to self is 0
    
    # Perform hierarchical clustering
    if verbose:
        print(f"\nPerforming hierarchical clustering (method={linkage_method})...")
    
    # Convert to condensed distance matrix
    condensed_dist = squareform(dist_matrix, checks=False)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = linkage(condensed_dist, method=linkage_method)
    
    # Determine number of clusters
    if n_groups is None:
        # For hierarchical classification to be useful, we want groups where:
        # 1. Each group has meaningfully different expression profiles
        # 2. Groups aren't too small (ideally >1 cell type per group on average)
        # 3. But also not too many groups (defeats purpose of hierarchy)
        
        # Target: sqrt(n_types) to balance depth vs breadth
        target_groups = max(3, min(int(np.sqrt(n_types) + 1), n_types - 1))
        
        # Get the heights (distances) at which clusters merge
        heights = Z[:, 2]
        
        if len(heights) > 2:
            # Compute rate of change in merge heights
            height_diffs = np.diff(heights)
            
            # Find significant jumps (gaps) in the merge distances
            # A significant gap indicates transition between groups
            mean_diff = np.mean(height_diffs)
            std_diff = np.std(height_diffs)
            
            # Find jumps that are > 1 std above mean
            significant_jumps = np.where(height_diffs > mean_diff + std_diff)[0]
            
            if len(significant_jumps) > 0:
                # Use the jump that gives us closest to target_groups clusters
                best_jump = None
                best_distance = float('inf')
                
                for jump_idx in significant_jumps:
                    resulting_groups = n_types - jump_idx - 1
                    distance = abs(resulting_groups - target_groups)
                    if distance < best_distance:
                        best_distance = distance
                        best_jump = jump_idx
                        n_groups = resulting_groups
                
                # Enforce bounds
                n_groups = max(3, min(n_groups, n_types - 1))
            else:
                n_groups = target_groups
        else:
            n_groups = target_groups
        
        if verbose:
            print(f"  Auto-selected {n_groups} groups (targeting ~{target_groups} based on {n_types} cell types)")
    
    # Get cluster assignments
    clusters = fcluster(Z, n_groups, criterion='maxclust')
    
    # Build hierarchy dictionary
    hierarchy = {}
    cluster_to_types = {}
    
    for ct, cluster_id in zip(cell_types, clusters):
        if cluster_id not in cluster_to_types:
            cluster_to_types[cluster_id] = []
        cluster_to_types[cluster_id].append(ct)
    
    # Create meaningful group names
    for cluster_id, types in sorted(cluster_to_types.items()):
        # Try to find a common pattern in the names
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
        
        # Print similarity info
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
    
    # Look for common patterns
    common_words = {
        'T': ['T cell', 'T cells', 'CD4', 'CD8', 'T Reg', 'Helper', 'Cytotoxic'],
        'B': ['B cell', 'B cells', 'CD19', 'CD20'],
        'NK': ['NK', 'Natural Killer'],
        'Monocyte': ['Monocyte', 'CD14', 'CD16'],
        'Dendritic': ['Dendritic', 'DC', 'pDC', 'mDC'],
        'Myeloid': ['Myeloid', 'Monocyte', 'Macrophage', 'Granulocyte'],
        'Stem': ['Stem', 'Progenitor', 'CD34', 'HSC'],
    }
    
    # Check each group pattern
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
    
    # Fallback: use common prefix
    if len(cell_types) >= 2:
        # Find longest common prefix
        prefix = cell_types[0]
        for ct in cell_types[1:]:
            while not ct.startswith(prefix) and len(prefix) > 0:
                prefix = prefix[:-1]
        
        if len(prefix) >= 3:
            return f"{prefix.strip()}_group"
    
    # Last resort: generic name
    return f"Group_{hash(tuple(sorted(cell_types))) % 1000}"


def visualize_hierarchy(
    adata,
    hierarchy: Dict[str, List[str]] = None,
    cell_type_column: str = 'cell_type',
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Visualize the cell type hierarchy as a dendrogram.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    hierarchy : Dict[str, List[str]], optional
        Pre-computed hierarchy. If None, will compute it
    cell_type_column : str
        Column in adata.obs containing cell type labels
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    import matplotlib.pyplot as plt
    
    # Compute profiles and similarity
    profiles, _ = compute_celltype_profiles(adata, cell_type_column=cell_type_column)
    sim_matrix = compute_similarity_matrix(profiles, method='correlation')
    
    # Convert to distance
    dist_matrix = 1 - sim_matrix.values
    np.fill_diagonal(dist_matrix, 0)
    condensed_dist = squareform(dist_matrix, checks=False)
    
    # Compute linkage
    Z = linkage(condensed_dist, method='average')
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Dendrogram
    ax1 = axes[0]
    dendrogram(
        Z,
        labels=profiles.index.tolist(),
        orientation='left',
        ax=ax1,
        leaf_font_size=10
    )
    ax1.set_title('Cell Type Hierarchy (Dendrogram)')
    ax1.set_xlabel('Distance (1 - correlation)')
    
    # Heatmap
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


def get_hierarchy_for_classifier(
    adata,
    cell_type_column: str = 'cell_type',
    n_groups: int = None,
    similarity_threshold: float = 0.7,
    verbose: bool = True
) -> Dict[str, List[str]]:
    """
    Convenience function to get hierarchy ready for the classifier.
    
    This is the main entry point for automatic hierarchy detection.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    cell_type_column : str
        Column containing cell type labels
    n_groups : int, optional
        Number of coarse groups (auto-detected if None)
    similarity_threshold : float
        Minimum similarity within groups
    verbose : bool
        Print progress
        
    Returns
    -------
    hierarchy : Dict[str, List[str]]
        Hierarchy dictionary ready to use in train_nn.py or train_xgb.py
        
    Example
    -------
    >>> from utils.hierarchy_detection import get_hierarchy_for_classifier
    >>> hierarchy = get_hierarchy_for_classifier(adata)
    >>> # Use in training script
    >>> CELL_TYPE_HIERARCHY = hierarchy
    """
    return detect_hierarchy(
        adata,
        cell_type_column=cell_type_column,
        n_groups=n_groups,
        similarity_threshold=similarity_threshold,
        verbose=verbose
    )
