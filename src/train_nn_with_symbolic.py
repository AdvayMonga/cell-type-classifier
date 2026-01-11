"""
Train neural network with symbolic regression-enhanced features.

This script demonstrates PHASE 3: Training with augmented features.

Usage:
    # Baseline (no symbolic regression)
    python train_nn_with_symbolic.py --data data/PBMC/pbmc68k_processed_5k.h5ad

    # With symbolic regression
    python train_nn_with_symbolic.py --data data/PBMC/pbmc68k_processed_5k.h5ad --use-symbolic

    # With symbolic regression + hyperparameter tuning
    python train_nn_with_symbolic.py --data data/PBMC/pbmc68k_processed_5k.h5ad --use-symbolic --tune
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import os
import pickle

# Import existing modules
from models import CellTypeClassifier, train_model, evaluate_model
from utils import load_and_validate_data

# Import new symbolic regression modules
from symbolic_regression import (
    get_marker_genes_per_celltype,
    discover_gene_relationships,
    compute_regulatory_features,
    create_augmented_feature_matrix,
    print_discovered_equations,
    plot_equation_fits,
    visualize_regulatory_network
)


def load_gene_expression_and_features(adata, use_unsmoothed=True):
    """Load gene expression and velocity features."""
    
    # Gene expression
    if use_unsmoothed and 'unsmoothed_log' in adata.layers:
        gene_expr = adata.layers['unsmoothed_log']
    else:
        gene_expr = adata.X
    
    if hasattr(gene_expr, 'toarray'):
        gene_expr = gene_expr.toarray()
    
    # Velocity features
    velocity_features = [
        'velocity_pseudotime', 'latent_time', 'velocity_confidence',
        'velocity_magnitude', 'S_score', 'G2M_score',
        'mean_expression', 'expression_variance', 'n_genes_expressed'
    ]
    
    available_features = [f for f in velocity_features if f in adata.obs.columns]
    if len(available_features) > 0:
        velocity_data = adata.obs[available_features].values
    else:
        velocity_data = np.zeros((adata.shape[0], 0))
    
    return gene_expr, velocity_data


def train_baseline_model(adata, args):
    """Train baseline model WITHOUT symbolic regression."""
    
    print(f"\n{'='*60}")
    print(f"BASELINE: Training without symbolic regression")
    print(f"{'='*60}\n")
    
    # Load features
    gene_expr, velocity_data = load_gene_expression_and_features(adata, use_unsmoothed=True)
    
    # Combine features
    X = np.concatenate([gene_expr, velocity_data], axis=1)
    y = adata.obs['cell_type'].values
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Features: {X.shape[1]} (genes: {gene_expr.shape[1]}, velocity: {velocity_data.shape[1]})")
    print(f"Samples: {X.shape[0]}")
    print(f"Classes: {len(label_encoder.classes_)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Create model
    model = CellTypeClassifier(
        input_size=X.shape[1],
        num_classes=len(label_encoder.classes_),
        dropout=0.3
    )
    
    # Train
    trained_model = train_model(
        model, train_loader, test_loader,
        learning_rate=0.0001,
        weight_decay=0.001,
        patience=5,
        max_epochs=100
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("BASELINE EVALUATION")
    print("="*60 + "\n")
    
    y_true_train, y_pred_train = evaluate_model(trained_model, train_loader)
    y_true_test, y_pred_test = evaluate_model(trained_model, test_loader)
    
    train_acc = accuracy_score(y_true_train, y_pred_train)
    test_acc = accuracy_score(y_true_test, y_pred_test)
    
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Overfitting gap: {(train_acc - test_acc):.4f}")
    
    print("\nPer-class performance:")
    print(classification_report(
        y_true_test, y_pred_test,
        target_names=label_encoder.classes_,
        digits=3
    ))
    
    return {
        'model': trained_model,
        'label_encoder': label_encoder,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'y_true_test': y_true_test,
        'y_pred_test': y_pred_test
    }


def train_symbolic_enhanced_model(adata, args):
    """Train model WITH symbolic regression features."""
    
    print(f"\n{'='*60}")
    print(f"SYMBOLIC REGRESSION ENHANCED MODEL")
    print(f"{'='*60}\n")
    
    # PHASE 1: Discover gene relationships
    print("PHASE 1: Discovering gene regulatory relationships...")
    
    marker_dict = get_marker_genes_per_celltype(
        adata,
        cell_type_column='cell_type',
        n_markers=args.n_markers,
        method='predefined'  # Use biologically known markers
    )
    
    # Discover relationships using symbolic regression
    models = discover_gene_relationships(
        adata,
        marker_dict,
        cell_type_column='cell_type',
        use_unsmoothed=True,
        min_cells_per_type=100,
        niterations=args.pysr_iterations,
        timeout_in_seconds=args.pysr_timeout
    )
    
    # Save discovered equations
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VIZ_DIR = os.path.join(PROJECT_ROOT, 'visualizations')
    os.makedirs(VIZ_DIR, exist_ok=True)
    
    equations_file = os.path.join(VIZ_DIR, 'discovered_equations.txt')
    print_discovered_equations(models, top_n=3, save_path=equations_file)
    
    # Visualize some relationships
    if len(models) > 0:
        # Plot first cell type's first gene
        first_cell_type = list(models.keys())[0]
        first_gene = list(models[first_cell_type].keys())[0]
        plot_equation_fits(adata, models, first_cell_type, first_gene, save_dir=VIZ_DIR)
        visualize_regulatory_network(models, first_cell_type, 
                                     save_path=os.path.join(VIZ_DIR, f'{first_cell_type}_network.png'))
    
    # PHASE 2: Compute regulatory features
    print("\nPHASE 2: Computing regulatory features...")
    
    regulatory_features = compute_regulatory_features(
        adata, models, use_unsmoothed=True
    )
    
    # Create augmented feature matrix
    X_augmented = create_augmented_feature_matrix(
        adata,
        regulatory_features,
        use_unsmoothed=True,
        include_velocity_features=True,
        gene_subset=None  # Use all genes
    )
    
    # PHASE 3: Train classifier with augmented features
    print("\nPHASE 3: Training classifier with augmented features...")
    
    X = X_augmented.values
    y = adata.obs['cell_type'].values
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\nAugmented feature matrix:")
    print(f"  Total features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Classes: {len(label_encoder.classes_)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Create model
    model = CellTypeClassifier(
        input_size=X.shape[1],
        num_classes=len(label_encoder.classes_),
        dropout=0.3
    )
    
    # Train
    trained_model = train_model(
        model, train_loader, test_loader,
        learning_rate=0.0001,
        weight_decay=0.001,
        patience=5,
        max_epochs=100
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("SYMBOLIC-ENHANCED MODEL EVALUATION")
    print("="*60 + "\n")
    
    y_true_train, y_pred_train = evaluate_model(trained_model, train_loader)
    y_true_test, y_pred_test = evaluate_model(trained_model, test_loader)
    
    train_acc = accuracy_score(y_true_train, y_pred_train)
    test_acc = accuracy_score(y_true_test, y_pred_test)
    
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Overfitting gap: {(train_acc - test_acc):.4f}")
    
    print("\nPer-class performance:")
    print(classification_report(
        y_true_test, y_pred_test,
        target_names=label_encoder.classes_,
        digits=3
    ))
    
    return {
        'model': trained_model,
        'label_encoder': label_encoder,
        'symbolic_models': models,
        'regulatory_features': regulatory_features,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'y_true_test': y_true_test,
        'y_pred_test': y_pred_test
    }


def compare_models(baseline_results, symbolic_results):
    """Compare baseline vs symbolic-enhanced models."""
    
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON")
    print(f"{'='*60}\n")
    
    print(f"{'Metric':<30} {'Baseline':<15} {'Symbolic':<15} {'Δ':<15}")
    print("-" * 75)
    
    # Accuracy comparison
    baseline_test = baseline_results['test_acc']
    symbolic_test = symbolic_results['test_acc']
    delta_test = symbolic_test - baseline_test
    
    print(f"{'Test Accuracy':<30} {baseline_test:<15.4f} {symbolic_test:<15.4f} {delta_test:+.4f}")
    
    baseline_train = baseline_results['train_acc']
    symbolic_train = symbolic_results['train_acc']
    
    print(f"{'Train Accuracy':<30} {baseline_train:<15.4f} {symbolic_train:<15.4f} {symbolic_train - baseline_train:+.4f}")
    
    baseline_gap = baseline_train - baseline_test
    symbolic_gap = symbolic_train - symbolic_test
    
    print(f"{'Overfitting Gap':<30} {baseline_gap:<15.4f} {symbolic_gap:<15.4f} {symbolic_gap - baseline_gap:+.4f}")
    
    # Statistical significance (optional)
    from scipy import stats
    
    # Confusion matrices
    baseline_cm = confusion_matrix(baseline_results['y_true_test'], baseline_results['y_pred_test'])
    symbolic_cm = confusion_matrix(symbolic_results['y_true_test'], symbolic_results['y_pred_test'])
    
    print(f"\n{'='*60}")
    print(f"INTERPRETATION")
    print(f"{'='*60}\n")
    
    if delta_test > 0.01:
        print(f"✓ Symbolic regression IMPROVED test accuracy by {delta_test*100:.2f}%")
    elif delta_test < -0.01:
        print(f"✗ Symbolic regression DECREASED test accuracy by {abs(delta_test)*100:.2f}%")
    else:
        print(f"≈ Symbolic regression had MINIMAL effect on test accuracy ({delta_test*100:.2f}%)")
    
    if symbolic_gap < baseline_gap:
        print(f"✓ Symbolic regression REDUCED overfitting by {(baseline_gap - symbolic_gap)*100:.2f}%")
    else:
        print(f"✗ Symbolic regression INCREASED overfitting by {(symbolic_gap - baseline_gap)*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Train cell type classifier with symbolic regression')
    parser.add_argument('--data', type=str, required=True, help='Path to processed .h5ad file')
    parser.add_argument('--use-symbolic', action='store_true', 
                       help='Use symbolic regression features (default: baseline only)')
    parser.add_argument('--compare', action='store_true',
                       help='Train both baseline and symbolic models and compare')
    parser.add_argument('--n-markers', type=int, default=7,
                       help='Number of marker genes per cell type (default: 7)')
    parser.add_argument('--pysr-iterations', type=int, default=40,
                       help='PySR iterations (default: 40, increase for better equations)')
    parser.add_argument('--pysr-timeout', type=int, default=300,
                       help='PySR timeout in seconds per model (default: 300)')
    parser.add_argument('--tune', action='store_true',
                       help='Perform hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    adata = load_and_validate_data(args.data)
    
    if args.compare or (not args.use_symbolic):
        # Train baseline model
        baseline_results = train_baseline_model(adata, args)
    
    if args.compare or args.use_symbolic:
        # Train symbolic-enhanced model
        symbolic_results = train_symbolic_enhanced_model(adata, args)
    
    if args.compare:
        # Compare both models
        compare_models(baseline_results, symbolic_results)
        
        # Save both models
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
        
        torch.save(baseline_results['model'].state_dict(), 
                  os.path.join(MODELS_DIR, 'neural_network_baseline.pt'))
        torch.save(symbolic_results['model'].state_dict(),
                  os.path.join(MODELS_DIR, 'neural_network_symbolic.pt'))
        
        # Save symbolic models for reuse
        with open(os.path.join(MODELS_DIR, 'symbolic_regression_models.pkl'), 'wb') as f:
            pickle.dump(symbolic_results['symbolic_models'], f)
        
        print(f"\n✓ Models saved to {MODELS_DIR}/")


if __name__ == '__main__':
    main()
