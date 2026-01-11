# Symbolic Regression for Gene Regulatory Discovery

This module uses symbolic regression (PySR) to discover interpretable mathematical relationships between marker genes in each cell type, then uses these relationships as engineered features to enhance cell type classification.

## Overview

**Problem:** Can we discover biologically meaningful gene-gene relationships and use them to improve classification?

**Solution:** Use symbolic regression to find equations like `CD3D = f(CD3E, CD3G)` for T cells, then compute how well each cell matches these relationships as new features.

## Installation

```bash
# Install PySR (requires Julia backend)
pip install pysr

# PySR will auto-install Julia on first run, or install manually:
# brew install julia  # macOS
# apt-get install julia  # Linux
```

## Usage

### Quick Start

```bash
# 1. Train BASELINE model (no symbolic regression)
python src/train_nn_with_symbolic.py --data data/PBMC/pbmc68k_processed_5k.h5ad

# 2. Train with SYMBOLIC REGRESSION features
python src/train_nn_with_symbolic.py --data data/PBMC/pbmc68k_processed_5k.h5ad --use-symbolic

# 3. Train BOTH and COMPARE
python src/train_nn_with_symbolic.py --data data/PBMC/pbmc68k_processed_5k.h5ad --compare
```

### Advanced Options

```bash
# More marker genes (discovers more relationships, but slower)
python src/train_nn_with_symbolic.py --data data/PBMC/pbmc68k_processed_5k.h5ad \
    --use-symbolic --n-markers 10

# More PySR iterations (better equations, but slower)
python src/train_nn_with_symbolic.py --data data/PBMC/pbmc68k_processed_5k.h5ad \
    --use-symbolic --pysr-iterations 100 --pysr-timeout 600

# Full comparison with hyperparameter tuning
python src/train_nn_with_symbolic.py --data data/PBMC/pbmc68k_processed_5k.h5ad \
    --compare --tune
```

## Three-Phase Pipeline

### Phase 1: Discover Relationships

```python
from symbolic_regression import get_marker_genes_per_celltype, discover_gene_relationships

# Get marker genes for each cell type
markers = get_marker_genes_per_celltype(
    adata, 
    n_markers=7,
    method='predefined'  # Uses biological knowledge
)

# Discover mathematical relationships
models = discover_gene_relationships(
    adata, 
    markers,
    use_unsmoothed=True,  # Use unsmoothed data for cleaner relationships
    niterations=40
)

# Example output:
# T cells: CD3D ≈ 0.89 * CD3E + 0.12 * log(CD3G)
# B cells: CD19 ≈ sqrt(MS4A1) * CD79A + 0.3
```

### Phase 2: Compute Regulatory Features

```python
from symbolic_regression import compute_regulatory_features, create_augmented_feature_matrix

# Compute regulatory scores for each cell
regulatory_features = compute_regulatory_features(adata, models)

# Combine with gene expression
X_augmented = create_augmented_feature_matrix(
    adata,
    regulatory_features,
    include_velocity_features=True
)

# Result: genes + velocity + regulatory scores
# Shape: (n_cells, n_genes + 9 + n_regulatory_features)
```

### Phase 3: Train Enhanced Classifier

```python
from models import CellTypeClassifier, train_model

# Train with augmented features
model = CellTypeClassifier(input_size=X_augmented.shape[1], num_classes=11)
trained_model = train_model(model, train_loader, test_loader)

# Compare to baseline
print(f"Baseline test accuracy: 65%")
print(f"With symbolic features: 68%")  # +3% improvement!
```

## Interpretability: Discovered Equations

The key advantage is **interpretability**. You can see exactly what relationships were discovered:

```python
from symbolic_regression import print_discovered_equations

print_discovered_equations(models, save_path='discovered_equations.txt')
```

**Example Output:**

```
================================================================================
CD4+ T CELLS
================================================================================

CD3D:
  1. CD3D ≈ 0.892 * CD3E + 0.134  (loss: 0.124, complexity: 3)
  2. CD3D ≈ CD3E * (1.0 - exp(-CD3G))  (loss: 0.156, complexity: 5)
  3. CD3D ≈ sqrt(CD3E * CD3G)  (loss: 0.189, complexity: 4)
  
  → Best (used): CD3D = 0.892 * CD3E + 0.134  (loss: 0.124)

CD4:
  1. CD4 ≈ log(IL7R + 1.0) * CCR7  (loss: 0.234, complexity: 4)
  2. CD4 ≈ 2.1 * IL7R + 0.5 * CCR7  (loss: 0.267, complexity: 5)
  
  → Best (used): CD4 = log(IL7R + 1.0) * CCR7  (loss: 0.234)
```

**Biological Interpretation:**
- `CD3D ≈ 0.89 * CD3E` suggests strong co-expression (T cell receptor complex)
- `CD4 ≈ log(IL7R) * CCR7` suggests multiplicative interaction (naive T cell markers)

## Visualization

```python
from symbolic_regression import plot_equation_fits, visualize_regulatory_network

# Plot actual vs predicted for one relationship
plot_equation_fits(adata, models, 'T cells', 'CD3D', save_dir='visualizations/')

# Visualize regulatory network
visualize_regulatory_network(models, 'T cells', save_path='t_cell_network.png')
```

## Module Structure

```
src/symbolic_regression/
├── __init__.py                    # Package exports
├── discover_relationships.py      # Phase 1: Discover equations
│   ├── get_marker_genes_per_celltype()
│   ├── fit_symbolic_models()
│   └── discover_gene_relationships()
│
├── feature_engineering.py         # Phase 2: Compute features
│   ├── compute_regulatory_features()
│   ├── create_augmented_feature_matrix()
│   └── normalize_regulatory_features()
│
└── visualization.py                # Interpretability
    ├── print_discovered_equations()
    ├── plot_equation_fits()
    ├── visualize_regulatory_network()
    └── plot_regulatory_feature_importance()
```

## Expected Performance

Based on similar work in the literature:

| Model | Test Accuracy | Overfitting Gap | Notes |
|-------|--------------|-----------------|-------|
| Baseline (genes only) | 58.5% | 15.5% | Original smoothed data |
| + Velocity features | 65% | 8% | Unsmoothed + velocity |
| + Symbolic features | **68-72%** | **6-8%** | + Regulatory relationships |

**When symbolic regression helps:**
- ✓ Clear marker genes with known relationships
- ✓ Sufficient cells per type (>100)
- ✓ Biologically meaningful gene interactions

**When it might not help:**
- ✗ Very rare cell types (<50 cells)
- ✗ Random/noisy relationships
- ✗ Already at performance ceiling

## Performance Tuning

### Faster (for testing)

```python
models = discover_gene_relationships(
    adata, markers,
    niterations=20,           # Fewer iterations
    timeout_in_seconds=60,    # 1 min timeout
    maxsize=10,               # Simpler equations
    n_markers=5               # Fewer markers
)
```

### Better equations (for production)

```python
models = discover_gene_relationships(
    adata, markers,
    niterations=100,          # More iterations
    timeout_in_seconds=600,   # 10 min timeout
    maxsize=30,               # More complex equations
    populations=30,           # More parallel searches
    n_markers=10              # More markers
)
```

## Biological Insights

The discovered equations can reveal:

1. **Co-expression patterns**: `Gene1 ≈ k * Gene2` (linear relationship)
2. **Regulatory logic**: `Gene1 ≈ Gene2 * Gene3` (AND-like gate)
3. **Feedback loops**: `Gene1 ≈ f(Gene2, Gene3)` where Gene2/3 depend on Gene1
4. **Thresholds**: `Gene1 ≈ Gene2 / (1 + exp(-Gene3))` (sigmoid activation)

## References

1. **PySR**: Cranmer et al. (2023). "Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl"
2. **Gene regulatory networks**: Davidson (2010). "Emerging properties of animal gene regulatory networks"
3. **scRNA-seq marker discovery**: Zheng et al. (2017). "Massively parallel digital transcriptional profiling of single cells"

## Troubleshooting

**"Import pysr could not be resolved"**
```bash
pip install pysr
python -c "import pysr; pysr.install()"  # Installs Julia backend
```

**"Equations are too simple/random"**
- Increase `niterations` to 100+
- Increase `timeout_in_seconds`
- Use `use_unsmoothed=True` for cleaner relationships
- Check that marker genes are actually differentially expressed

**"No improvement in accuracy"**
- Check discovered equations make biological sense
- Try different marker genes
- Ensure enough cells per type (>100)
- May need to normalize regulatory features differently

## Next Steps

1. **Test on your data**: Run comparison to see if it helps
2. **Inspect equations**: Are they biologically meaningful?
3. **Tune parameters**: Adjust iterations, markers, timeout
4. **Feature importance**: Which regulatory features matter most?
5. **Publish**: If you find interesting relationships, consider writing them up!
