# Cell Type Classifier with RNA Velocity

A machine learning pipeline for classifying cell types from single-cell RNA sequencing data using gene expression and RNA velocity features. Applied to the PBMC68k dataset.

## Overview

This project builds a neural network classifier that predicts cell types (T cells, B cells, NK cells, monocytes, dendritic cells, stem cells) from single-cell transcriptomic data using scVelo for RNA velocity analysis. 

## Project Structure

```
cell-type-classifier/
├── data/PBMC/
│   ├── pbmc68k.h5ad                        # Raw data (65,877 cells × 33,939 genes)
│   ├── pbmc68k_processed.h5ad              # Processed data
│
├── src/
│   ├── models/                             # Model architectures
│   │   ├── cell_classifier.py              # Neural network definition
│   │   └── training.py                     # Training utilities
│   │
│   ├── utils/                              # Utility functions
│   │   └── data_loading.py                 # Data loading, feature extraction
│   │
│   ├── symbolic_regression/                 # Gene relationship discovery (optional)
│   │   ├── discover_relationships.py       # PySR-based discovery
│   │   ├── feature_engineering.py          # Regulatory features
│   │   └── visualization.py                # Network visualization
│   │
│   ├── prep_data.py                        # Pipeline Step 1: Preprocessing
│   ├── select_genes.py                     # Pipeline Step 2: Gene selection
│   ├── train_nn.py                         # Pipeline Step 3: Training (main)
│   ├── train_lr.py                         # Alternative: Logistic regression
│   ├── train_rf.py                         # Alternative: Random forest
│   └── visualize.py                        # Visualization tools
│
├── models/                                  # Saved model checkpoints
└── visualizations/                          # Plots and figures
```

### Architecture Design

**Pipeline Scripts** (orchestration, run these):
- `prep_data.py` - Preprocessing with RNA velocity (single self-contained file)
- `select_genes.py` - Selects informative genes
- `train_nn.py` - **Main training script** with interactive prompts

**Core Modules** (reusable components):
- `models/` - Model architectures and training loops
- `symbolic_regression/` - Optional gene relationship discovery

**Utilities** (helper functions):
- `utils/data_loading.py` - All data loading and feature extraction functions
  - `load_and_validate_data()` - Load .h5ad files
  - `load_gene_expression_and_features()` - Extract genes + velocity features
  - `add_symbolic_regression_features()` - Add discovered regulatory features

## Setup & Installation

### 1. Clone and Create Environment
```bash
git clone https://github.com/AdvayMonga/cell-type-classifier.git
cd cell-type-classifier

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Data
The PBMC68k dataset will be automatically downloaded on first run, or manually:
```python
import scvelo as scv
adata = scv.datasets.pbmc68k()
adata.write('data/PBMC/pbmc68k.h5ad')
```

## Usage

### Step 1: Preprocessing
```bash
# Run full preprocessing with RNA velocity (may take time)
python src/prep_data.py

# With velocity plots
python src/prep_data.py --plot
```

**Note:** The preprocessing always computes velocity features and saves them.
You can choose whether to use them during training via interactive prompts.

### Step 2: Gene Selection
```bash
# Select top 5k genes and force include specific markers
python src/select_genes.py --data data/PBMC/pbmc68k_processed.h5ad
```

### Step 3: Train Hierarchical Classifier
```bash
# Train hierarchical classifier
python src/train_nn.py --data data/PBMC/pbmc68k_processed_5k.h5ad
```

**Hierarchical Classification:**
- **Level 1**: Coarse classification (T cells vs B cells vs NK vs Myeloid vs Dendritic)
  - Uses gene expression only
- **Level 2**: Fine-grained classification within each group
  - Uses genes + boosted velocity features (2x boost)

**Interactive prompt:**
- `Include symbolic regression features? (y/n):` - Discover gene relationships (optional)

**Note:** Symbolic regression requires additional setup:
```bash
pip install pysr scipy networkx
python -c "import pysr; pysr.install()"
```

**Output:**
- `models/hierarchical_level1.pt` - Coarse classifier
- `models/hierarchical_level2.pt` - Fine-grained classifiers for each group

## References

1. **RNA Velocity**: La Manno et al. (2018). "RNA velocity of single cells." *Nature*, 560(7719), 494-498.
2. **scVelo**: Bergen et al. (2020). "Generalizing RNA velocity to transient cell states through dynamical modeling." *Nature Biotechnology*, 38(12), 1408-1414.
3. **PBMC68k Dataset**: Zheng et al. (2017). "Massively parallel digital transcriptional profiling of single cells." *Nature Communications*, 8, 14049.