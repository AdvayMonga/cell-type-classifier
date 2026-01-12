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
from utils import load_and_validate_data, load_gene_expression_and_features, add_symbolic_regression_features


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
    
    # Interactive prompts for feature selection
    print("\n" + "="*70)
    use_velocity = input("Include velocity features? (y/n): ").strip().lower() == 'y'
    use_symbolic = input("Include symbolic regression features? (y/n): ").strip().lower() == 'y'
    print("="*70)
    
    # Load features
    X = load_gene_expression_and_features(adata, use_velocity=use_velocity)
    y = adata.obs['cell_type'].values
    
    # Add symbolic regression features if requested
    if use_symbolic:
        X = add_symbolic_regression_features(adata, X)
    else:
        print("\n✓ Skipping symbolic regression")
    
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
