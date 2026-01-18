"""Hierarchical neural network training for cell type classification."""
try:
    import juliacall
except ImportError:
    pass

import numpy as np
import os
import torch
import argparse
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

from models import CellTypeClassifier, train_model, evaluate_model
from models import tune_architecture, create_model_from_params, get_training_params_from_study
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
        return velocity_array * boost_factor
    return None


def train_level_model(X_train, y_train, X_test, y_test, level_name, num_epochs=20,
                      verbose=True, use_smote=False, tune=False, n_trials=30):
    """Train a model for one level of the hierarchy."""

    if use_smote:
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique, counts))
        min_samples = counts.min()

        if verbose:
            print(f"\n  Class distribution before SMOTE:")
            for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
                print(f"    {cls}: {cnt}")

        if min_samples >= 6:
            target_min = 500
            sampling_strategy = {}
            for cls, cnt in class_counts.items():
                if cnt < target_min:
                    sampling_strategy[cls] = target_min

            if sampling_strategy:
                smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                if verbose:
                    print(f"\n  Applied SMOTE")

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    input_size = X_train.shape[1]
    num_classes = len(label_encoder.classes_)

    best_params = None
    if tune:
        if verbose:
            print(f"\n  Running Optuna tuning ({n_trials} trials)...")

        best_params, study = tune_architecture(
            X_train, y_train_encoded, num_classes,
            n_trials=n_trials, n_folds=3, num_epochs=10, device='cpu', verbose=verbose
        )

        model = create_model_from_params(best_params, input_size, num_classes)
        training_params = get_training_params_from_study(best_params)
        learning_rate = training_params['learning_rate']
        weight_decay = training_params['weight_decay']
        batch_size = training_params['batch_size']

        if verbose:
            n_layers = best_params['n_layers']
            hidden_sizes = [best_params[f'hidden_size_{i}'] for i in range(n_layers)]
            print(f"\n  Tuned architecture: {input_size} -> {hidden_sizes} -> {num_classes}")
    else:
        hidden_sizes = CellTypeClassifier.get_architecture_for_input_size(input_size)
        model = CellTypeClassifier(input_size, num_classes, hidden_sizes, dropout_rate=0.3)
        learning_rate = 0.0001
        weight_decay = 0.001
        batch_size = 32

        if verbose:
            print(f"\n  Model: {input_size} -> {hidden_sizes} -> {num_classes}")

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train_encoded))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test_encoded))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_model(model, train_loader, test_loader, num_epochs=num_epochs,
                learning_rate=learning_rate, weight_decay=weight_decay, patience=5, verbose=verbose)

    test_labels, test_preds = evaluate_model(model, test_loader)
    accuracy = accuracy_score(test_labels, test_preds)

    return model, label_encoder, accuracy, (test_labels, test_preds), best_params


def main():
    parser = argparse.ArgumentParser(description='Train hierarchical neural network')
    parser.add_argument('--data', type=str, required=True, help='Path to processed .h5ad file')
    parser.add_argument('--tune', action='store_true', help='Enable Optuna tuning')
    args = parser.parse_args()

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(PROJECT_ROOT, args.data) if not os.path.isabs(args.data) else args.data
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)

    adata = load_and_validate_data(data_path)

    print(f"\nCell types: {adata.obs['cell_type'].nunique()}")
    print(adata.obs['cell_type'].value_counts())

    print("\n" + "="*70)
    print("AUTO-DETECTING HIERARCHY")
    print("="*70)
    CELL_TYPE_HIERARCHY = get_hierarchy_for_classifier(adata, n_groups=None, verbose=True)

    print("\n" + "="*70)
    use_symbolic = input("Include symbolic regression features? (y/n): ").strip().lower() == 'y'
    print("="*70)

    print("\n" + "="*70)
    print("LOADING FEATURES")
    print("="*70)
    X_genes = load_gene_expression_and_features(adata, use_velocity=False)

    X_velocity = load_velocity_features(adata, boost_factor=2.0)
    if X_velocity is not None:
        print(f"\nVelocity features: {X_velocity.shape}")

    X_symbolic = None
    if use_symbolic:
        X_temp = load_gene_expression_and_features(adata, use_velocity=True)
        X_combined = add_symbolic_regression_features(adata, X_temp)
        n_symbolic = X_combined.shape[1] - X_temp.shape[1]
        if n_symbolic > 0:
            X_symbolic = X_combined[:, -n_symbolic:]
            print(f"\nSymbolic features: {n_symbolic}")

    y_fine = adata.obs['cell_type'].values
    y_coarse = np.array([get_coarse_label(ct, CELL_TYPE_HIERARCHY) for ct in y_fine])

    unknown_mask = y_coarse == 'Unknown'
    if unknown_mask.sum() > 0:
        print(f"\nFiltering {unknown_mask.sum()} unknown cells")
        X_genes = X_genes[~unknown_mask]
        if X_velocity is not None:
            X_velocity = X_velocity[~unknown_mask]
        if X_symbolic is not None:
            X_symbolic = X_symbolic[~unknown_mask]
        y_fine = y_fine[~unknown_mask]
        y_coarse = y_coarse[~unknown_mask]

    indices = np.arange(len(y_fine))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_fine)

    print(f"\n{'='*70}")
    print(f"DATASET: {len(train_idx)} train, {len(test_idx)} test")
    print(f"{'='*70}")

    # Level 1: Coarse classification
    print(f"\n{'='*70}")
    print(f"LEVEL 1: COARSE CLASSIFICATION")
    print(f"{'='*70}")

    X_level1 = X_genes

    model_level1, encoder_level1, acc_level1, (l1_labels, l1_preds), l1_params = train_level_model(
        X_level1[train_idx], y_coarse[train_idx],
        X_level1[test_idx], y_coarse[test_idx],
        level_name="Level 1", num_epochs=20, verbose=True, tune=args.tune
    )

    print(f"\nLevel 1 Accuracy: {acc_level1:.1%}")
    print(classification_report(l1_labels, l1_preds, target_names=encoder_level1.classes_))

    # Level 2: Fine-grained classification
    print(f"\n{'='*70}")
    print(f"LEVEL 2: FINE-GRAINED CLASSIFICATION")
    print(f"{'='*70}")

    X_level2_base = X_genes.copy()
    if X_velocity is not None:
        X_level2_base = np.column_stack([X_level2_base, X_velocity])
    if X_symbolic is not None:
        X_level2_base = np.column_stack([X_level2_base, X_symbolic])

    print(f"Level 2 features: {X_level2_base.shape[1]}")

    level2_models = {}
    level2_encoders = {}
    level2_accuracies = {}

    for group_name, subtypes in CELL_TYPE_HIERARCHY.items():
        if len(subtypes) <= 1:
            print(f"\n  Skipping {group_name} (single subtype)")
            continue

        print(f"\n  {'─'*60}")
        print(f"  {group_name}: {subtypes}")

        group_mask_train = np.isin(y_fine[train_idx], subtypes)
        group_mask_test = np.isin(y_fine[test_idx], subtypes)

        n_train, n_test = group_mask_train.sum(), group_mask_test.sum()
        if n_train < 50 or n_test < 10:
            print(f"  Skipping (train={n_train}, test={n_test})")
            continue

        X_group_train = X_level2_base[train_idx][group_mask_train]
        y_group_train = y_fine[train_idx][group_mask_train]
        X_group_test = X_level2_base[test_idx][group_mask_test]
        y_group_test = y_fine[test_idx][group_mask_test]

        use_smote = (group_name == 'T_cells')
        model, encoder, accuracy, (labels, preds), l2_params = train_level_model(
            X_group_train, y_group_train, X_group_test, y_group_test,
            level_name=f"Level 2 ({group_name})", num_epochs=25,
            verbose=True, use_smote=use_smote, tune=args.tune
        )

        level2_models[group_name] = model
        level2_encoders[group_name] = encoder
        level2_accuracies[group_name] = accuracy

        print(f"\n  {group_name} Accuracy: {accuracy:.1%}")
        print(classification_report(labels, preds, target_names=encoder.classes_))

    # Overall accuracy
    print(f"\n{'='*70}")
    print(f"HIERARCHICAL RESULTS")
    print(f"{'='*70}")

    correct = 0
    total = len(test_idx)

    for idx in test_idx:
        true_fine = y_fine[idx]
        true_coarse = y_coarse[idx]

        with torch.no_grad():
            model_level1.eval()
            x_l1 = torch.FloatTensor(X_level1[idx:idx+1])
            pred_coarse_idx = model_level1(x_l1).argmax(dim=1).item()
            pred_coarse = encoder_level1.classes_[pred_coarse_idx]

        if pred_coarse != true_coarse:
            continue

        if pred_coarse in level2_models:
            with torch.no_grad():
                model_l2 = level2_models[pred_coarse]
                encoder_l2 = level2_encoders[pred_coarse]
                model_l2.eval()
                x_l2 = torch.FloatTensor(X_level2_base[idx:idx+1])
                pred_fine_idx = model_l2(x_l2).argmax(dim=1).item()
                pred_fine = encoder_l2.classes_[pred_fine_idx]

            if pred_fine == true_fine:
                correct += 1
        else:
            correct += 1

    overall_accuracy = correct / total

    print(f"\nLevel 1: {acc_level1:.1%}")
    for group_name, acc in level2_accuracies.items():
        print(f"Level 2 ({group_name}): {acc:.1%}")

    print(f"\n" + "="*70)
    print(f"OVERALL: {overall_accuracy:.1%}")
    print(f"="*70)

    # Save models
    print(f"\nSaving models...")

    if l1_params is not None:
        n_layers = l1_params['n_layers']
        hidden_sizes = [l1_params[f'hidden_size_{i}'] for i in range(n_layers)]
        use_attention = l1_params.get('use_attention', True)
        attention_hidden = l1_params.get('attention_hidden', 128)
    else:
        hidden_sizes = CellTypeClassifier.get_architecture_for_input_size(X_level1.shape[1])
        use_attention = True
        attention_hidden = 128

    level1_data = {
        'model_state_dict': model_level1.state_dict(),
        'label_encoder': encoder_level1,
        'input_size': X_level1.shape[1],
        'hidden_sizes': hidden_sizes,
        'num_classes': len(encoder_level1.classes_),
        'accuracy': acc_level1,
        'use_attention': use_attention,
        'attention_hidden': attention_hidden,
        'tuned_params': l1_params,
    }
    torch.save(level1_data, os.path.join(MODELS_DIR, 'hierarchical_level1.pt'))

    level2_data = {
        'models': {g: m.state_dict() for g, m in level2_models.items()},
        'encoders': level2_encoders,
        'accuracies': level2_accuracies,
        'input_size': X_level2_base.shape[1],
        'hierarchy': CELL_TYPE_HIERARCHY,
    }
    torch.save(level2_data, os.path.join(MODELS_DIR, 'hierarchical_level2.pt'))

    print(f"Saved to models/hierarchical_level1.pt and models/hierarchical_level2.pt")
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
