# Cell Type Classifier

A machine learning pipeline for classifying cell types from single-cell RNA sequencing data using gene expression profiles and RNA velocity features. Applied to pbmc68k data set from scVelo.

## Overview

This project builds a neural network classifier that predicts cell types (T cells, B cells, NK cells, monocytes, etc.) from single-cell transcriptomic data. Key innovations include integrating RNA velocity dynamics, implementing a two-stage preprocessing pipeline for rapid experimentation, and systematic hyperparameter optimization to address overfitting in high-dimensional biological data.


## Project Structure
```
cell-type-classifier/
├── data/
│   └── PBMC/
│       ├── pbmc68k.h5ad              # Raw data (65,877 cells)
│       ├── pbmc68k_processed.h5ad    # Processed data
├── src/
│   ├── velocity_data.py              # Preprocessing with velocity features
│   ├── select_genes.py               # Fast gene selection for experimentation
│   ├── train_nn.py                   # Neural network training
│   ├── train_lr.py                   # Logistic regression 
│   ├── train_rf.py                   # Random forest
│   ├── inspect_data.py               # Dataset inspection
│   └── visualize.py                  # Generate plots and analysis
├── models/                           # Saved models
├── visualizations/                   # UMAP, velocity stream, QC plots
├── requirements.txt                  # Dependencies
└── README.md
```

## Setup & Usage
```bash
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```