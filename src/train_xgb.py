"""
Hierarchical classifier with XGBoost for Level 2 (fine-grained classification).

This script trains a 2-level hierarchical classifier:
- Level 1: Neural network for coarse classification (automatically detected from data)
- Level 2: XGBoost for fine-grained classification (better with imbalanced classes)

XGBoost is used for Level 2 because it handles class imbalance better than neural networks.
The hierarchy is automatically detected by clustering similar cell types together.

Usage:
    python src/train_xgb.py --data data/PBMC/pbmc68k_processed_5k.h5ad
"""
# Import juliacall before torch to avoid segfault (PySR uses Julia)
try:
    import juliacall
except ImportError:
    pass

import numpy as np
import os
import torch
import argparse
import xgboost as xgb
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Import model modules
from models import CellTypeClassifier, train_model, evaluate_model
from utils import load_and_validate_data, load_gene_expression_and_features, add_symbolic_regression_features
from utils import get_hierarchy_for_classifier


def get_coarse_label(fine_label, hierarchy):
    """Map fine-grained cell type to coarse group."""
    for coarse, fine_types in hierarchy.items():
        if fine_label in fine_types:
            return coarse
    return 'Unknown'


def load_velocity_features(adata, boost_factor=2.0):
    """Load velocity features with additional boosting for Level 2."""
    feature_names = [
        'velocity_pseudotime', 'latent_time', 'velocity_confidence', 'velocity_magnitude',
        'S_score', 'G2M_score', 'mean_expression', 'expression_variance', 'n_genes_expressed'
    ]
    
    velocity_features = []
    for feat in feature_names:
        if feat in adata.obs.columns:
            values = adata.obs[feat].values.reshape(-1, 1)
            velocity_features.append(values)
    
    if velocity_features:
        velocity_array = np.column_stack(velocity_features)
        velocity_array = np.nan_to_num(velocity_array, nan=0.0, posinf=0.0, neginf=0.0)
        velocity_array = velocity_array * boost_factor
        return velocity_array
    return None


def train_level1_nn(X_train, y_train, X_test, y_test, num_epochs=20, verbose=True):
    """Train neural network for Level 1 (coarse classification)."""
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train_encoded)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test_encoded)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    input_size = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    hidden_sizes = CellTypeClassifier.get_architecture_for_input_size(input_size)
    
    if verbose:
        print(f"\n  Model: {input_size} inputs → {hidden_sizes} → {num_classes} classes")
    
    model = CellTypeClassifier(input_size, num_classes, hidden_sizes, dropout_rate=0.3)
    train_model(
        model, train_loader, test_loader,
        num_epochs=num_epochs,
        learning_rate=0.0001,
        weight_decay=0.001,
        patience=5,
        verbose=verbose
    )
    
    test_labels, test_preds = evaluate_model(model, test_loader)
    accuracy = accuracy_score(test_labels, test_preds)
    
    return model, label_encoder, accuracy, (test_labels, test_preds)


def train_level2_xgboost(X_train, y_train, X_test, y_test, group_name, verbose=True):
    """
    Train XGBoost classifier for Level 2 (fine-grained classification).
    
    XGBoost handles class imbalance better than neural networks through:
    - scale_pos_weight for binary classification
    - sample_weight for multi-class
    - Tree-based splits work well with few samples
    """
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    num_classes = len(label_encoder.classes_)
    
    if verbose:
        print(f"\n  Class distribution:")
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
            print(f"    {cls}: {cnt}")
    
    # Compute sample weights for imbalanced classes
    class_counts = np.bincount(y_train_encoded)
    total_samples = len(y_train_encoded)
    # Weight inversely proportional to class frequency
    class_weights = total_samples / (num_classes * class_counts)
    sample_weights = class_weights[y_train_encoded]
    
    if verbose:
        print(f"\n  Using class weights:")
        for i, cls in enumerate(label_encoder.classes_):
            print(f"    {cls}: {class_weights[i]:.3f}")
    
    # XGBoost parameters - handle binary vs multi-class
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    }
    
    # Set objective based on number of classes
    if num_classes == 2:
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'logloss'
    else:
        params['objective'] = 'multi:softprob'
        params['eval_metric'] = 'mlogloss'
        params['num_class'] = num_classes
    
    if verbose:
        print(f"\n  Training XGBoost: {X_train.shape[1]} features → {num_classes} classes")
    
    # Create and train XGBoost model with sample weights
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train_encoded,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test_encoded)],
        verbose=False
    )
    
    # Evaluate - predict returns class labels directly
    y_pred = model.predict(X_test)
    # Ensure y_pred is integer array (XGBoost may return float for binary)
    y_pred = y_pred.astype(int)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    
    return model, label_encoder, accuracy, (y_test_encoded, y_pred)


def main():
    parser = argparse.ArgumentParser(description='Train hierarchical classifier with XGBoost for Level 2')
    parser.add_argument('--data', type=str, required=True, help='Path to processed .h5ad file')
    args = parser.parse_args()

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(PROJECT_ROOT, args.data) if not os.path.isabs(args.data) else args.data
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Load data
    adata = load_and_validate_data(data_path)
    
    print(f"\nCell types found: {adata.obs['cell_type'].nunique()}")
    print(f"Cell type distribution:")
    print(adata.obs['cell_type'].value_counts())
    
    # Auto-detect hierarchy from data
    print("\n" + "="*70)
    print("AUTO-DETECTING HIERARCHY")
    print("="*70)
    CELL_TYPE_HIERARCHY = get_hierarchy_for_classifier(
        adata, 
        n_groups=None,  # Auto-determine optimal number of groups
        verbose=True
    )
    
    # Single prompt for symbolic regression
    print("\n" + "="*70)
    use_symbolic = input("Include symbolic regression features? (y/n): ").strip().lower() == 'y'
    print("="*70)
    
    # Load features
    print("\n" + "="*70)
    print("LOADING FEATURES")
    print("="*70)
    X_genes = load_gene_expression_and_features(adata, use_velocity=False)
    
    X_velocity = load_velocity_features(adata, boost_factor=2.0)
    if X_velocity is not None:
        print(f"\n✓ Loaded velocity features for Level 2")
        print(f"  Shape: {X_velocity.shape}")
    else:
        print(f"\n⚠ No velocity features found")
    
    X_symbolic = None
    if use_symbolic:
        X_temp = load_gene_expression_and_features(adata, use_velocity=True)
        X_combined = add_symbolic_regression_features(adata, X_temp)
        n_symbolic = X_combined.shape[1] - X_temp.shape[1]
        if n_symbolic > 0:
            X_symbolic = X_combined[:, -n_symbolic:]
            print(f"\n✓ Loaded {n_symbolic} symbolic regression features")
    else:
        print(f"\n✓ Skipping symbolic regression features")
    
    # Get labels
    y_fine = adata.obs['cell_type'].values
    y_coarse = np.array([get_coarse_label(ct, CELL_TYPE_HIERARCHY) for ct in y_fine])
    
    # Filter unknown cell types
    unknown_mask = y_coarse == 'Unknown'
    if unknown_mask.sum() > 0:
        print(f"\n⚠ Warning: {unknown_mask.sum()} cells have unknown cell types (excluded)")
        X_genes = X_genes[~unknown_mask]
        if X_velocity is not None:
            X_velocity = X_velocity[~unknown_mask]
        if X_symbolic is not None:
            X_symbolic = X_symbolic[~unknown_mask]
        y_fine = y_fine[~unknown_mask]
        y_coarse = y_coarse[~unknown_mask]
    
    # Train/test split
    indices = np.arange(len(y_fine))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y_fine
    )
    
    print(f"\n{'='*70}")
    print(f"DATASET SPLIT")
    print(f"{'='*70}")
    print(f"Training set: {len(train_idx)} cells")
    print(f"Test set: {len(test_idx)} cells")
    
    # ========================================================================
    # LEVEL 1: COARSE CLASSIFICATION (Neural Network)
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"LEVEL 1: COARSE CLASSIFICATION (Neural Network)")
    print(f"{'='*70}")
    print(f"Classes: {list(CELL_TYPE_HIERARCHY.keys())}")
    print(f"Features: Gene expression only ({X_genes.shape[1]} genes)")
    
    X_level1 = X_genes
    
    model_level1, encoder_level1, acc_level1, (l1_labels, l1_preds) = train_level1_nn(
        X_level1[train_idx], y_coarse[train_idx],
        X_level1[test_idx], y_coarse[test_idx],
        num_epochs=20,
        verbose=True
    )
    
    print(f"\n✓ Level 1 Accuracy: {acc_level1:.1%}")
    print(f"\nLevel 1 Classification Report:")
    print(classification_report(l1_labels, l1_preds, target_names=encoder_level1.classes_))
    
    # ========================================================================
    # LEVEL 2: FINE-GRAINED CLASSIFICATION (XGBoost)
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"LEVEL 2: FINE-GRAINED CLASSIFICATION (XGBoost)")
    print(f"{'='*70}")
    
    # Prepare Level 2 features
    X_level2_base = X_genes.copy()
    feature_info = [f"{X_genes.shape[1]} genes"]
    
    if X_velocity is not None:
        X_level2_base = np.column_stack([X_level2_base, X_velocity])
        feature_info.append(f"{X_velocity.shape[1]} velocity")
    
    if X_symbolic is not None:
        X_level2_base = np.column_stack([X_level2_base, X_symbolic])
        feature_info.append(f"{X_symbolic.shape[1]} symbolic")
    
    print(f"Features: {' + '.join(feature_info)}")
    print(f"Total Level 2 features: {X_level2_base.shape[1]}")
    
    level2_models = {}
    level2_encoders = {}
    level2_accuracies = {}
    
    for group_name, subtypes in CELL_TYPE_HIERARCHY.items():
        if len(subtypes) <= 1:
            print(f"\n  Skipping {group_name} (only 1 subtype: {subtypes[0]})")
            continue
        
        print(f"\n  {'─'*60}")
        print(f"  Training Level 2 XGBoost for: {group_name}")
        print(f"  {'─'*60}")
        print(f"  Subtypes: {subtypes}")
        
        group_mask_train = np.isin(y_fine[train_idx], subtypes)
        group_mask_test = np.isin(y_fine[test_idx], subtypes)
        
        n_train = group_mask_train.sum()
        n_test = group_mask_test.sum()
        
        if n_train < 50 or n_test < 10:
            print(f"  ⚠ Skipping {group_name}: insufficient samples")
            continue
        
        print(f"  Samples: {n_train} train, {n_test} test")
        
        X_group_train = X_level2_base[train_idx][group_mask_train]
        y_group_train = y_fine[train_idx][group_mask_train]
        X_group_test = X_level2_base[test_idx][group_mask_test]
        y_group_test = y_fine[test_idx][group_mask_test]
        
        model, encoder, accuracy, (labels, preds) = train_level2_xgboost(
            X_group_train, y_group_train,
            X_group_test, y_group_test,
            group_name=group_name,
            verbose=True
        )
        
        level2_models[group_name] = model
        level2_encoders[group_name] = encoder
        level2_accuracies[group_name] = accuracy
        
        print(f"\n  ✓ {group_name} Accuracy: {accuracy:.1%}")
        print(f"\n  Classification Report for {group_name}:")
        print(classification_report(labels, preds, target_names=encoder.classes_))
    
    # ========================================================================
    # OVERALL HIERARCHICAL ACCURACY
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"HIERARCHICAL CLASSIFICATION RESULTS (NN + XGBoost)")
    print(f"{'='*70}")
    
    correct = 0
    total = len(test_idx)
    
    for i, idx in enumerate(test_idx):
        true_fine = y_fine[idx]
        true_coarse = y_coarse[idx]
        
        # Level 1 prediction (Neural Network)
        with torch.no_grad():
            model_level1.eval()
            x_l1 = torch.FloatTensor(X_level1[idx:idx+1])
            pred_coarse_idx = model_level1(x_l1).argmax(dim=1).item()
            pred_coarse = encoder_level1.classes_[pred_coarse_idx]
        
        if pred_coarse != true_coarse:
            continue
        
        if pred_coarse in level2_models:
            # Level 2 prediction (XGBoost)
            model_l2 = level2_models[pred_coarse]
            encoder_l2 = level2_encoders[pred_coarse]
            x_l2 = X_level2_base[idx:idx+1]
            pred_fine_idx = model_l2.predict(x_l2)[0]
            pred_fine = encoder_l2.classes_[pred_fine_idx]
            
            if pred_fine == true_fine:
                correct += 1
        else:
            correct += 1
    
    overall_accuracy = correct / total
    
    print(f"\nPER-LEVEL ACCURACY:")
    print(f"  Level 1 (Neural Network): {acc_level1:.1%}")
    for group_name, acc in level2_accuracies.items():
        print(f"  Level 2 XGBoost ({group_name}): {acc:.1%}")
    
    print(f"\n" + "="*70)
    print(f"OVERALL HIERARCHICAL ACCURACY: {overall_accuracy:.1%}")
    print(f"="*70)
    
    # ========================================================================
    # SAVE MODELS
    # ========================================================================
    print(f"\nSaving models...")
    
    # Save Level 1 (PyTorch)
    level1_data = {
        'model_state_dict': model_level1.state_dict(),
        'label_encoder': encoder_level1,
        'input_size': X_level1.shape[1],
        'hidden_sizes': CellTypeClassifier.get_architecture_for_input_size(X_level1.shape[1]),
        'num_classes': len(encoder_level1.classes_),
        'accuracy': acc_level1,
    }
    torch.save(level1_data, os.path.join(MODELS_DIR, 'xgb_hierarchical_level1.pt'))
    print(f"  ✓ Saved Level 1 model to models/xgb_hierarchical_level1.pt")
    
    # Save Level 2 (XGBoost)
    import pickle
    level2_data = {
        'models': level2_models,
        'encoders': level2_encoders,
        'accuracies': level2_accuracies,
        'input_size': X_level2_base.shape[1],
        'hierarchy': CELL_TYPE_HIERARCHY,
    }
    with open(os.path.join(MODELS_DIR, 'xgb_hierarchical_level2.pkl'), 'wb') as f:
        pickle.dump(level2_data, f)
    print(f"  ✓ Saved Level 2 XGBoost models to models/xgb_hierarchical_level2.pkl")
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
