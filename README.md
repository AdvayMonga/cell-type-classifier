# Cell Type Classifier with RNA Velocity

A machine learning pipeline for classifying cell types from single-cell RNA sequencing data using gene expression and RNA velocity features. Applied to the PBMC68k dataset.

## Overview

This project builds a neural network classifier that predicts cell types (T cells, B cells, NK cells, monocytes, dendritic cells, stem cells) from single-cell transcriptomic data using scVelo for RNA velocity analysis. 

## Project Structure

```
cell-type-classifier/
├── data/PBMC/
│   ├── pbmc68k.h5ad                        # Raw data (65,877 cells × 33,939 genes)
│   ├── pbmc68k_processed.h5ad              # Processed with velocity features
│   └── pbmc68k_processed_5k.h5ad           # 5k selected genes
│
├── src/
│   ├── preprocessing/                       # Modular preprocessing
│   ├── models/                              # Neural network modules
│   ├── utils/                               # Utility functions
│   ├── symbolic_regression/                 # Gene relationship discovery (optional)
│   │
│   ├── velocity_data.py                    # Main preprocessing pipeline
│   ├── select_genes.py                     # Gene selection
│   ├── train_nn.py                         # Main training script
│   ├── train_lr.py                         # Logistic regression
│   ├── train_rf.py                         # Random forest
│   └── visualize.py                        # Visualization tools
│
├── models/                                  # Saved model checkpoints
└── visualizations/                          # Plots and figures
```

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
# Run full preprocessing (~15 min)
python src/velocity_data.py

# With velocity plots
python src/velocity_data.py --plot
```

### Step 2: Gene Selection
```bash
# Select top 5k genes and force include specific markers
python src/select_genes.py --data data/PBMC/pbmc68k_processed.h5ad
```

### Step 3: Train Classifier
```bash
# Train with default hyperparameters
python src/train_nn.py --data data/PBMC/pbmc68k_processed_5k.h5ad

# Train with hyperparameter tuning
python src/train_nn.py --data data/PBMC/pbmc68k_processed_5k.h5ad --tune
```

### Optional: Symbolic Regression

For discovering interpretable gene relationships:

```bash
# Install dependencies
pip install pysr scipy networkx
python -c "import pysr; pysr.install()"

# Train with symbolic regression features
python src/train_nn_with_symbolic.py --data data/PBMC/pbmc68k_processed_5k.h5ad --use-symbolic

# Compare baseline vs symbolic
python src/train_nn_with_symbolic.py --data data/PBMC/pbmc68k_processed_5k.h5ad --compare
```

See `src/symbolic_regression/README.md` for details.

## References

1. **RNA Velocity**: La Manno et al. (2018). "RNA velocity of single cells." *Nature*, 560(7719), 494-498.
2. **scVelo**: Bergen et al. (2020). "Generalizing RNA velocity to transient cell states through dynamical modeling." *Nature Biotechnology*, 38(12), 1408-1414.
3. **PBMC68k Dataset**: Zheng et al. (2017). "Massively parallel digital transcriptional profiling of single cells." *Nature Communications*, 8, 14049.