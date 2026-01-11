"""
Quick example: Using symbolic regression to enhance cell type classification.

This script shows a minimal working example of the three-phase pipeline.
"""

import scanpy as sc
import numpy as np
from symbolic_regression import (
    get_marker_genes_per_celltype,
    discover_gene_relationships,
    compute_regulatory_features,
    create_augmented_feature_matrix,
    print_discovered_equations
)

# Load your processed data
adata = sc.read_h5ad('data/PBMC/pbmc68k_processed_5k.h5ad')

print("="*80)
print("SYMBOLIC REGRESSION QUICKSTART EXAMPLE")
print("="*80)

# ============================================================================
# PHASE 1: Discover Gene Regulatory Relationships
# ============================================================================

print("\nPHASE 1: Discovering relationships...")

# Get marker genes for each cell type
markers = get_marker_genes_per_celltype(
    adata,
    cell_type_column='cell_type',
    n_markers=5,  # Use 5 markers per cell type (fast)
    method='predefined'  # Use biological knowledge
)

# Discover relationships with symbolic regression
# NOTE: This is CPU-intensive! First run may take 5-10 minutes
models = discover_gene_relationships(
    adata,
    markers,
    use_unsmoothed=True,  # Use unsmoothed data
    min_cells_per_type=100,
    niterations=20,  # Reduced for speed (use 40+ for production)
    timeout_in_seconds=60  # 1 minute per equation
)

# Print discovered equations
print_discovered_equations(models, top_n=2)

# ============================================================================
# PHASE 2: Compute Regulatory Features
# ============================================================================

print("\nPHASE 2: Computing regulatory features...")

# Compute regulatory scores
regulatory_features = compute_regulatory_features(adata, models)

print(f"\nRegulatory features shape: {regulatory_features.shape}")
print(f"Features: {list(regulatory_features.columns[:5])}...")

# Create augmented feature matrix
X_augmented = create_augmented_feature_matrix(
    adata,
    regulatory_features,
    use_unsmoothed=True,
    include_velocity_features=True
)

print(f"\nAugmented matrix shape: {X_augmented.shape}")
print(f"  Genes: {adata.n_vars}")
print(f"  Velocity: 9")
print(f"  Regulatory: {regulatory_features.shape[1]}")
print(f"  Total: {X_augmented.shape[1]}")

# ============================================================================
# PHASE 3: Use in Classifier
# ============================================================================

print("\nPHASE 3: Ready for training!")
print("\nNext steps:")
print("1. Train baseline model (genes + velocity):")
print("   python src/train_nn.py --data data/PBMC/pbmc68k_processed_5k.h5ad")
print("\n2. Train symbolic-enhanced model:")
print("   python src/train_nn_with_symbolic.py --data data/PBMC/pbmc68k_processed_5k.h5ad --use-symbolic")
print("\n3. Compare both models:")
print("   python src/train_nn_with_symbolic.py --data data/PBMC/pbmc68k_processed_5k.h5ad --compare")

# ============================================================================
# Example: Use regulatory features directly
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE: Examining regulatory features")
print("="*80)

# Look at one cell's regulatory scores
cell_idx = 0
cell_scores = regulatory_features.iloc[cell_idx]
print(f"\nCell {cell_idx} ({adata.obs['cell_type'].iloc[cell_idx]}):")
print(f"Top 5 regulatory scores:")
top_features = cell_scores.abs().sort_values(ascending=False).head(5)
for feature, score in top_features.items():
    print(f"  {feature}: {score:.3f}")

# Compare cells from different types
print("\nComparing T cell vs B cell:")
t_cells = adata.obs['cell_type'].str.contains('T')
b_cells = adata.obs['cell_type'].str.contains('B')

if t_cells.sum() > 0 and b_cells.sum() > 0:
    t_cell_scores = regulatory_features[t_cells].mean()
    b_cell_scores = regulatory_features[b_cells].mean()
    
    print("\nMean regulatory scores:")
    print(f"  T cell scores: {t_cell_scores.mean():.3f}")
    print(f"  B cell scores: {b_cell_scores.mean():.3f}")

print("\n✓ Example complete!")
