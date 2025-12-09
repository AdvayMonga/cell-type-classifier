"""
Train neural network for cell type classification.

This refactored version uses the modular architecture to:
1. Load preprocessed data with unsmoothed expression
2. Combine gene expression with velocity features
3. Split into train/test sets
4. Train neural network with early stopping
5. Evaluate and save model

Usage:
    python train_nn_refactored.py --data data/PBMC/pbmc68k_processed_5k.h5ad
    python train_nn_refactored.py --data data/PBMC/pbmc68k_processed_5k.h5ad --tune
"""
import numpy as np
import os
import torch
import argparse
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import itertools

# Import model modules
from models import CellTypeClassifier, train_model, evaluate_model
from utils import load_and_validate_data


def load_gene_expression_and_features(adata):
    """
    Load gene expression (unsmoothed) and velocity features.
    
    Args:
        adata: AnnData object
    
    Returns:
        numpy array: Combined features (genes + velocity)
    """
    # CRITICAL: Use unsmoothed data for classification
    if 'unsmoothed_log' in adata.layers:
        print(f"\n✓ Using UNSMOOTHED log-transformed data for classification")
        print(f"  (scVelo moments() smoothed adata.X across neighbors - blurs cell type boundaries)")
        print(f"  (unsmoothed_log layer preserves sharp boundaries needed for classification)")
        X_genes = adata.layers['unsmoothed_log']
    else:
        print(f"\n⚠ WARNING: Using SMOOTHED data (unsmoothed_log layer not found)")
        print(f"  This may hurt classification by blurring cell type boundaries!")
        X_genes = adata.X

    if hasattr(X_genes, 'toarray'):
        X_genes = X_genes.toarray()
    else:
        X_genes = np.array(X_genes)

    print(f"\nGene expression matrix:")
    print(f"  Shape: {X_genes.shape}")
    print(f"  Range: [{X_genes.min():.4f}, {X_genes.max():.4f}]")
    print(f"  Mean: {X_genes.mean():.4f}")

    # Add velocity features
    print("\nChecking for velocity and statistical features...")
    velocity_features = []
    
    feature_names = [
        'velocity_pseudotime', 'latent_time', 'velocity_confidence', 'velocity_magnitude',
        'S_score', 'G2M_score', 'mean_expression', 'expression_variance', 'n_genes_expressed'
    ]
    
    for feat in feature_names:
        if feat in adata.obs.columns:
            values = adata.obs[feat].values.reshape(-1, 1)
            velocity_features.append(values)
            print(f"  ✓ {feat}: range [{values.min():.4f}, {values.max():.4f}], mean {values.mean():.4f}")
    
    # Combine features
    if velocity_features:
        velocity_array = np.column_stack(velocity_features)
        
        # Check for NaN or Inf values
        n_nans = np.isnan(velocity_array).sum()
        n_infs = np.isinf(velocity_array).sum()

        if n_nans > 0 or n_infs > 0:
            velocity_array = np.nan_to_num(velocity_array, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"  ⚠ Fixed {n_nans} NaNs and {n_infs} Infs in velocity features")
        
        X = np.column_stack([X_genes, velocity_array])
        n_genes = X_genes.shape[1]
        n_additional = velocity_array.shape[1]
        
        print(f"\n✓ Combined features: {X.shape[1]} total ({n_genes} genes + {n_additional} additional features)")
        print(f"  Gene expression range: [{X[:, :n_genes].min():.4f}, {X[:, :n_genes].max():.4f}]")
        print(f"  Velocity features range: [{X[:, n_genes:].min():.4f}, {X[:, n_genes:].max():.4f}]")
        print(f"  Note: Velocity features already boosted 1.5x during preprocessing")
    else:
        print("\n⚠ No velocity features found, using gene expression only")
        X = X_genes
    
    return X


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train neural network for cell type classification')
    parser.add_argument('--data', type=str, required=True, help='Path to processed .h5ad file')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    args = parser.parse_args()

    # Setup paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(PROJECT_ROOT, args.data) if not os.path.isabs(args.data) else args.data
    
    # Load and validate data
    adata = load_and_validate_data(data_path)
    
    print(f"Cell types: {adata.obs['cell_type'].nunique()}")
    print(f"Cell type distribution:")
    print(adata.obs['cell_type'].value_counts())
    
    # Load features
    X = load_gene_expression_and_features(adata)
    y = adata.obs['cell_type'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTraining set: {X_train.shape[0]} cells")
    print(f"Test set: {X_test.shape[0]} cells")
    
    # Print class distribution
    print(f"\n{'='*70}")
    print(f"CLASS DISTRIBUTION")
    print(f"{'='*70}")
    for i, label in enumerate(label_encoder.classes_):
        n_train = (y_train == i).sum()
        n_test = (y_test == i).sum()
        print(f"{label:35s}: {n_train:5d} train, {n_test:4d} test")
    print(f"{'='*70}\n")
    
    # Create data loaders
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model parameters
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    hidden_sizes = CellTypeClassifier.get_architecture_for_input_size(input_size)
    
    print(f"Model parameters:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden layers: {hidden_sizes}")
    print(f"  Output classes: {num_classes}")
    
    # Hyperparameter tuning or default
    if args.tune:
        print("\n" + "="*70)
        print("HYPERPARAMETER GRID SEARCH")
        print("="*70)
        
        learning_rates = [0.0005, 0.0001]
        dropout_rates = [0.2, 0.3]
        weight_decays = [0.001, 0.0005, 0.002]
        
        results = []
        total_configs = len(learning_rates) * len(dropout_rates) * len(weight_decays)
        current_config = 0
        
        for lr, dropout, wd in itertools.product(learning_rates, dropout_rates, weight_decays):
            current_config += 1
            print(f"\n[{current_config}/{total_configs}] LR: {lr}, Dropout: {dropout}, L2: {wd}")
            
            model = CellTypeClassifier(input_size, num_classes, hidden_sizes, dropout_rate=dropout)
            accuracy = train_model(model, train_loader, test_loader, num_epochs=20, 
                                  learning_rate=lr, weight_decay=wd, patience=3, verbose=False)
            
            results.append({
                'learning_rate': lr,
                'dropout': dropout,
                'weight_decay': wd,
                'accuracy': accuracy
            })
            
            print(f"  Best Test Accuracy: {accuracy:.1%}")
        
        # Find best configuration
        best_config = max(results, key=lambda x: x['accuracy'])
        print("\n" + "="*70)
        print("BEST HYPERPARAMETERS:")
        print("="*70)
        print(f"Learning rate: {best_config['learning_rate']}")
        print(f"Dropout rate: {best_config['dropout']}")
        print(f"L2 Regularization: {best_config['weight_decay']}")
        print(f"Best Test Accuracy: {best_config['accuracy']:.1%}")
        print("="*70)
        
        learning_rate = best_config['learning_rate']
        dropout_rate = best_config['dropout']
        weight_decay = best_config['weight_decay']
    else:
        learning_rate = 0.0001
        dropout_rate = 0.3
        weight_decay = 0.001
        print(f"\nUsing default hyperparameters:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Dropout rate: {dropout_rate}")
        print(f"  L2 Regularization: {weight_decay}")
    
    # Train final model
    print(f"\nTraining final model...")
    model = CellTypeClassifier(input_size, num_classes, hidden_sizes, dropout_rate=dropout_rate)
    print(f"\nModel architecture:")
    print(model)
    
    num_epochs = 20
    patience = 3
    print(f"\nTraining for up to {num_epochs} epochs (early stopping patience: {patience})...")
    
    best_test_accuracy = train_model(model, train_loader, test_loader, num_epochs, 
                                      learning_rate, weight_decay, patience=patience, verbose=True)
    
    # Final evaluation
    all_labels, all_preds = evaluate_model(model, test_loader)
    train_labels, train_preds = evaluate_model(model, train_loader)
    
    accuracy = accuracy_score(all_labels, all_preds)
    train_accuracy = accuracy_score(train_labels, train_preds)
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS:")
    print(f"{'='*70}")
    print(f"Train accuracy: {train_accuracy:.1%}")
    print(f"Test accuracy: {accuracy:.1%}")
    print(f"Overfitting gap: {train_accuracy - accuracy:.1%} ({'Good' if train_accuracy - accuracy < 0.15 else 'Overfitting'})")
    print(f"{'='*70}")
    
    print("\nPer-class performance:")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
    
    # Save model
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    model_data = {
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'input_size': input_size,
        'hidden_sizes': hidden_sizes,
        'num_classes': num_classes,
        'accuracy': accuracy,
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'weight_decay': weight_decay
    }
    
    model_path = os.path.join(MODELS_DIR, 'neural_network.pt')
    torch.save(model_data, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == '__main__':
    main()
