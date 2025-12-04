# Cell Type Classifier

Project to classify cell types from single-cell gene expression data. Trying out logistic regression, Random Forest, PyTorch Neural Network.
Went more in-depth with the neural network, tuning hyperparameters and testing out different network architectures. Performed RNA Velocity analysis and included velocity features in neural network training.

## Overview

This project takes single-cell RNA sequencing data and predicts the cell type (e.g., T cell, B cell, macrophage) based on gene expression and RNA Velocity

## Project Structure
```
cell-type-classifier/
├── data/               # Gene expression datasets
├── notebooks/          # Jupyter notebooks for exploration
├── src/                # scripts
├── requirements.txt    # dependencies
└── README.md
```

## Setup
```bash
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```
