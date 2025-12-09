"""Normalize and boost velocity features for classification."""
import numpy as np


def normalize_and_boost_features(adata, boost_factor=1.5):
    """
    Normalize all velocity features to 0-1 range, then boost by a factor.
    
    This helps balance velocity features (which would be 0-1) with gene expression
    (which is 0-4.38 after log normalization). Boosting to 0-1.5 gives velocity
    features more influence in the classifier.
    
    Args:
        adata: AnnData object with features in .obs
        boost_factor: Multiplier after 0-1 normalization (default 1.5x)
    
    Returns:
        AnnData object with normalized and boosted features
    """
    print("\nNormalizing and boosting velocity features...")
    
    velocity_features = [
        'velocity_pseudotime', 'latent_time', 'velocity_confidence', 
        'velocity_magnitude', 'S_score', 'G2M_score', 
        'mean_expression', 'expression_variance', 'n_genes_expressed'
    ]
    
    print(f"   Boost factor: {boost_factor}x (to balance with gene expression 0-4.38 range)")
    
    for feat in velocity_features:
        if feat not in adata.obs.columns:
            continue
            
        values = adata.obs[feat].values
        
        # Check for NaNs before normalization
        nan_count = np.isnan(values).sum()
        if nan_count > 0:
            print(f"   ❌ ERROR: {feat} contains {nan_count} NaNs! Replacing with 0...")
            values = np.nan_to_num(values, nan=0.0)
        
        # Normalize to 0-1 range
        val_min = values.min()
        val_max = values.max()
        val_range = val_max - val_min
        
        if val_range < 1e-10:
            # All values are identical
            print(f"   ⚠️  {feat} has no variance (constant: {val_min:.4f}), setting to 0.5")
            normalized = np.full_like(values, 0.5)
        else:
            normalized = (values - val_min) / val_range
        
        # Check for NaNs after normalization
        nan_count_after = np.isnan(normalized).sum()
        if nan_count_after > 0:
            print(f"   ❌ ERROR: {feat} produced {nan_count_after} NaNs after normalization!")
            normalized = np.nan_to_num(normalized, nan=0.0)
        
        # BOOST by factor to increase influence relative to genes
        boosted = normalized * boost_factor
        
        adata.obs[feat] = boosted
        print(f"   ✓ Normalized & boosted {feat}: range [{boosted.min():.4f}, {boosted.max():.4f}], mean={boosted.mean():.4f}")
    
    return adata
