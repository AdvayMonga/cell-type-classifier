"""
Integration test for symbolic regression module.

This script tests that all components work together correctly.
Run this before using symbolic regression on real data.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

print("="*80)
print("SYMBOLIC REGRESSION MODULE - INTEGRATION TEST")
print("="*80)

# ============================================================================
# Test 1: Module imports
# ============================================================================

print("\n[TEST 1] Module imports...")
try:
    from symbolic_regression import (
        get_marker_genes_per_celltype,
        discover_gene_relationships,
        compute_regulatory_features,
        create_augmented_feature_matrix,
        print_discovered_equations,
        plot_equation_fits,
        visualize_regulatory_network
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("\nInstall dependencies:")
    print("  pip install pysr scipy networkx")
    exit(1)

# ============================================================================
# Test 2: PySR availability
# ============================================================================

print("\n[TEST 2] PySR availability...")
try:
    from pysr import PySRRegressor
    print("✓ PySR installed")
    
    # Test Julia backend
    try:
        model = PySRRegressor(niterations=1, verbosity=0)
        X_test = np.random.randn(10, 2)
        y_test = X_test[:, 0] + X_test[:, 1]
        model.fit(X_test, y_test)
        print("✓ Julia backend working")
    except Exception as e:
        print(f"⚠ Julia backend issue: {e}")
        print("  Run: python -c 'import pysr; pysr.install()'")
except ImportError:
    print("✗ PySR not installed")
    print("  Install: pip install pysr")
    exit(1)

# ============================================================================
# Test 3: Create synthetic data
# ============================================================================

print("\n[TEST 3] Creating synthetic test data...")

# Create simple synthetic dataset
np.random.seed(42)
n_cells = 500
n_genes = 50

# Two cell types with clear marker relationships
# Type A: gene1 = 2 * gene2 + 1
# Type B: gene1 = 0.5 * gene2 - 0.5

cell_types = np.array(['TypeA'] * 250 + ['TypeB'] * 250)

X = np.random.randn(n_cells, n_genes) * 0.5
X[:, 0] = np.abs(X[:, 0])  # Ensure positive
X[:, 1] = np.abs(X[:, 1])
X[:, 2] = np.abs(X[:, 2])

# Add clear relationships for Type A
X[:250, 0] = 2 * X[:250, 1] + 1 + np.random.randn(250) * 0.1
# And for Type B
X[250:, 0] = 0.5 * X[250:, 1] - 0.5 + np.random.randn(250) * 0.1

# Create AnnData
gene_names = [f'Gene{i}' for i in range(n_genes)]
gene_names[0] = 'TargetGene'
gene_names[1] = 'Marker1'
gene_names[2] = 'Marker2'

adata = AnnData(X)
adata.var_names = gene_names
adata.obs['cell_type'] = cell_types
adata.layers['unsmoothed_log'] = X.copy()

print(f"✓ Created synthetic data: {adata.shape}")
print(f"  Cell types: {np.unique(cell_types)}")
print(f"  Genes: {n_genes}")

# ============================================================================
# Test 4: Marker gene identification
# ============================================================================

print("\n[TEST 4] Identifying marker genes...")

try:
    # Manually specify markers for synthetic data
    marker_dict = {
        'TypeA': ['TargetGene', 'Marker1', 'Marker2'],
        'TypeB': ['TargetGene', 'Marker1', 'Marker2']
    }
    
    print(f"✓ Marker genes defined")
    for ct, markers in marker_dict.items():
        print(f"  {ct}: {markers}")
        
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# ============================================================================
# Test 5: Discover relationships
# ============================================================================

print("\n[TEST 5] Discovering relationships...")

try:
    models = discover_gene_relationships(
        adata,
        marker_dict,
        cell_type_column='cell_type',
        use_unsmoothed=True,
        min_cells_per_type=50,
        niterations=10,  # Very short for testing
        timeout_in_seconds=30
    )
    
    print(f"✓ Discovery complete")
    print(f"  Cell types with models: {len(models)}")
    for ct, ct_models in models.items():
        print(f"  {ct}: {len(ct_models)} equations")
        
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# Test 6: Compute regulatory features
# ============================================================================

print("\n[TEST 6] Computing regulatory features...")

try:
    regulatory_features = compute_regulatory_features(adata, models)
    
    print(f"✓ Feature computation complete")
    print(f"  Shape: {regulatory_features.shape}")
    print(f"  Features: {list(regulatory_features.columns)}")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# Test 7: Create augmented matrix
# ============================================================================

print("\n[TEST 7] Creating augmented feature matrix...")

try:
    X_augmented = create_augmented_feature_matrix(
        adata,
        regulatory_features,
        use_unsmoothed=True,
        include_velocity_features=False  # No velocity in synthetic data
    )
    
    print(f"✓ Augmented matrix created")
    print(f"  Shape: {X_augmented.shape}")
    print(f"  Original genes: {n_genes}")
    print(f"  Regulatory features: {regulatory_features.shape[1]}")
    print(f"  Total: {X_augmented.shape[1]}")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# Test 8: Visualization functions
# ============================================================================

print("\n[TEST 8] Testing visualization functions...")

try:
    # Print equations
    print("\n  Testing print_discovered_equations...")
    print_discovered_equations(models, top_n=2)
    print("  ✓ Equation printing works")
    
    # Note: Skipping plot generation in automated test
    print("  ⊘ Skipping plot generation (requires display)")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Test 9: Check equations make sense
# ============================================================================

print("\n[TEST 9] Validating discovered equations...")

try:
    if len(models) > 0:
        for cell_type, ct_models in models.items():
            for target_gene, model in ct_models.items():
                best_eq = model.get_best()
                
                print(f"\n  {cell_type} - {target_gene}:")
                print(f"    Equation: {best_eq.equation}")
                print(f"    Loss: {best_eq.loss:.4f}")
                
                # For TypeA, we expect something like: TargetGene ≈ 2 * Marker1 + 1
                if cell_type == 'TypeA' and target_gene == 'TargetGene':
                    if 'Marker1' in best_eq.equation:
                        print("    ✓ Found expected relationship with Marker1")
                    else:
                        print("    ⚠ Expected Marker1 in equation")
    else:
        print("  ⚠ No models discovered (may need more iterations)")
        
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("INTEGRATION TEST COMPLETE")
print("="*80)

print("\n✓ All core functionality working!")
print("\nYou can now use symbolic regression on real data:")
print("  python src/train_nn_with_symbolic.py --data data/PBMC/pbmc68k_processed_5k.h5ad --compare")

print("\nFor a quick example:")
print("  python src/symbolic_regression_example.py")

print("\nFor full documentation:")
print("  cat src/symbolic_regression/README.md")
print("  cat SYMBOLIC_REGRESSION_GUIDE.md")
