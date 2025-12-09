"""Compute RNA velocity features."""
import scvelo as scv
import scanpy as sc


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
