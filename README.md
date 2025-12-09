# Cell Type Classifier with RNA Velocity

A modular machine learning pipeline for classifying cell types from single-cell RNA sequencing data using gene expression profiles and RNA velocity features. Applied to the PBMC68k dataset from scVelo.

## Overview

This project builds a neural network classifier that predicts cell types (T cells, B cells, NK cells, monocytes, dendritic cells, stem cells) from single-cell transcriptomic data. 

## Project Structure

```
cell-type-classifier/
├── data/
│   └── PBMC/
│       ├── pbmc68k.h5ad                    # Raw data (65,877 cells × 33,939 genes)
│       ├── pbmc68k_processed.h5ad          # Processed (~10k genes, with unsmoothed layer)
│       └── pbmc68k_processed_5k.h5ad       # 5k selected genes + forced markers
│
├── src/
│   ├── preprocessing/                       # Modular preprocessing
│   │   ├── __init__.py
│   │   ├── filter_genes.py                 # Gene filtering with protected markers
│   │   ├── compute_velocity.py             # RNA velocity + save unsmoothed data
│   │   ├── extract_features.py             # 9 velocity & expression features
│   │   └── normalize_features.py           # Normalize 0-1, boost 1.5x
│   │
│   ├── models/                              # Neural network modules
│   │   ├── __init__.py
│   │   ├── cell_classifier.py              # Auto-sizing MLP architecture
│   │   └── training.py                     # Training with early stopping
│   │
│   ├── utils/                               # Utility functions
│   │   ├── __init__.py
│   │   └── data_loading.py                 # Data loading & validation
│   │
│   ├── velocity_data.py                    # Main preprocessing pipeline
│   ├── train_nn.py                         # Main training script
│   ├── select_genes.py                     # Fast gene selection
│   │
│   ├── train_lr.py                         # Logistic regression
│   ├── train_rf.py                         # Random forest
│   ├── inspect_data.py                     # Dataset inspection
│   └── visualize.py                        # Generate plots
│
├── models/                                  # Saved model checkpoints
├── visualizations/                          # UMAP, velocity stream, QC plots
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
└── REFACTORING.md                           # 🆕 Detailed refactoring docs
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

#### Step 1: Preprocessing
```bash
# Run full preprocessing (~15 min)
python src/velocity_data_refactored.py

# With velocity plots
python src/velocity_data_refactored.py --plot
```

#### Step 2: Gene Selection
```bash
# Select n genes and force include specific markers
python src/select_genes.py --data data/PBMC/pbmc68k_processed.h5ad
```

#### Step 3: Train Classifier
```bash
# Train with default hyperparameters
python src/train_nn_refactored.py --data data/PBMC/pbmc68k_processed_5k.h5ad

# Train with hyperparameter tuning
python src/train_nn_refactored.py --data data/PBMC/pbmc68k_processed_5k.h5ad --tune
```

## References

1. **RNA Velocity**: La Manno et al. (2018). "RNA velocity of single cells." *Nature*, 560(7719), 494-498.
2. **scVelo**: Bergen et al. (2020). "Generalizing RNA velocity to transient cell states through dynamical modeling." *Nature Biotechnology*, 38(12), 1408-1414.
3. **PBMC68k Dataset**: Zheng et al. (2017). "Massively parallel digital transcriptional profiling of single cells." *Nature Communications*, 8, 14049.