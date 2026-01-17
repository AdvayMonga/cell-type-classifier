"""Optuna-based hyperparameter tuning for neural network architecture."""
import optuna
from optuna.trial import Trial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
from typing import Optional, Dict, Any, Tuple

from .cell_classifier import CellTypeClassifier


def create_model_from_trial(trial: Trial, input_size: int, num_classes: int) -> CellTypeClassifier:
    """
    Create a CellTypeClassifier with hyperparameters suggested by Optuna trial.

    Args:
        trial: Optuna trial object
        input_size: Number of input features
        num_classes: Number of output classes

    Returns:
        CellTypeClassifier with trial-suggested hyperparameters
    """
    # Number of hidden layers (1-4)
    n_layers = trial.suggest_int('n_layers', 1, 4)

    # Hidden layer sizes (each layer can be 32-512 neurons)
    hidden_sizes = []
    for i in range(n_layers):
        # Each subsequent layer is typically smaller or equal
        if i == 0:
            max_size = 512
        else:
            max_size = hidden_sizes[i - 1]

        layer_size = trial.suggest_int(f'hidden_size_{i}', 32, max_size, step=32)
        hidden_sizes.append(layer_size)

    # Dropout rate
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)

    # Attention parameters
    use_attention = trial.suggest_categorical('use_attention', [True, False])
    attention_hidden = trial.suggest_int('attention_hidden', 64, 256, step=32) if use_attention else 128

    return CellTypeClassifier(
        input_size=input_size,
        num_classes=num_classes,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate,
        use_attention=use_attention,
        attention_hidden=attention_hidden
    )


def create_objective(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    n_folds: int = 3,
    num_epochs: int = 15,
    device: str = 'cpu'
):
    """
    Create an Optuna objective function for hyperparameter optimization.

    Args:
        X: Feature matrix
        y: Encoded labels
        num_classes: Number of classes
        n_folds: Number of cross-validation folds
        num_epochs: Maximum epochs per trial
        device: Device to train on

    Returns:
        Objective function for Optuna
    """
    input_size = X.shape[1]

    def objective(trial: Trial) -> float:
        # Get training hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

        # Cross-validation
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.LongTensor(y_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.LongTensor(y_val)
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Create model with trial hyperparameters
            model = create_model_from_trial(trial, input_size, num_classes)
            model = model.to(device)

            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )

            # Train with early stopping
            best_val_acc = 0
            patience_counter = 0
            patience = 3

            for epoch in range(num_epochs):
                # Training
                model.train()
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                # Validation
                model.eval()
                val_preds = []
                val_labels = []
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(device)
                        outputs = model(batch_X)
                        preds = outputs.argmax(dim=1).cpu().numpy()
                        val_preds.extend(preds)
                        val_labels.extend(batch_y.numpy())

                val_acc = accuracy_score(val_labels, val_preds)

                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

                # Pruning: report intermediate value
                trial.report(val_acc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            fold_accuracies.append(best_val_acc)

        return np.mean(fold_accuracies)

    return objective


def tune_architecture(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    n_trials: int = 50,
    n_folds: int = 3,
    num_epochs: int = 15,
    device: str = 'cpu',
    verbose: bool = True,
    study_name: Optional[str] = None
) -> Tuple[Dict[str, Any], optuna.Study]:
    """
    Run Optuna hyperparameter optimization to find the best architecture.

    Args:
        X: Feature matrix
        y: Encoded labels (integers)
        num_classes: Number of classes
        n_trials: Number of Optuna trials
        n_folds: Number of cross-validation folds
        num_epochs: Maximum epochs per trial
        device: Device to train on ('cpu' or 'cuda')
        verbose: Print progress
        study_name: Optional name for the study

    Returns:
        Tuple of (best_params dict, optuna.Study)
    """
    if verbose:
        print(f"\n{'='*60}")
        print("OPTUNA ARCHITECTURE TUNING")
        print(f"{'='*60}")
        print(f"  Trials: {n_trials}")
        print(f"  CV Folds: {n_folds}")
        print(f"  Max epochs per trial: {num_epochs}")
        print(f"  Device: {device}")
        print(f"  Input features: {X.shape[1]}")
        print(f"  Classes: {num_classes}")
        print(f"{'='*60}\n")

    # Create study with pruning
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name=study_name or 'cell_classifier_tuning'
    )

    # Create objective
    objective = create_objective(
        X, y, num_classes,
        n_folds=n_folds,
        num_epochs=num_epochs,
        device=device
    )

    # Run optimization
    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    # Extract best parameters
    best_params = study.best_params
    best_value = study.best_value

    if verbose:
        print(f"\n{'='*60}")
        print("TUNING RESULTS")
        print(f"{'='*60}")
        print(f"  Best CV Accuracy: {best_value:.1%}")
        print(f"\n  Best Architecture:")

        # Reconstruct hidden sizes
        n_layers = best_params['n_layers']
        hidden_sizes = [best_params[f'hidden_size_{i}'] for i in range(n_layers)]
        print(f"    Hidden layers: {hidden_sizes}")
        print(f"    Dropout: {best_params['dropout_rate']}")
        print(f"    Use attention: {best_params['use_attention']}")
        if best_params['use_attention']:
            print(f"    Attention hidden: {best_params['attention_hidden']}")

        print(f"\n  Best Training Params:")
        print(f"    Learning rate: {best_params['learning_rate']:.6f}")
        print(f"    Weight decay: {best_params['weight_decay']:.6f}")
        print(f"    Batch size: {best_params['batch_size']}")
        print(f"{'='*60}\n")

    return best_params, study


def create_model_from_params(
    params: Dict[str, Any],
    input_size: int,
    num_classes: int
) -> CellTypeClassifier:
    """
    Create a CellTypeClassifier from a params dictionary (e.g., from tuning).

    Args:
        params: Dictionary of hyperparameters
        input_size: Number of input features
        num_classes: Number of output classes

    Returns:
        CellTypeClassifier instance
    """
    # Reconstruct hidden sizes
    n_layers = params['n_layers']
    hidden_sizes = [params[f'hidden_size_{i}'] for i in range(n_layers)]

    return CellTypeClassifier(
        input_size=input_size,
        num_classes=num_classes,
        hidden_sizes=hidden_sizes,
        dropout_rate=params['dropout_rate'],
        use_attention=params.get('use_attention', True),
        attention_hidden=params.get('attention_hidden', 128)
    )


def get_training_params_from_study(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract training parameters from best params dictionary.

    Args:
        params: Best parameters from Optuna study

    Returns:
        Dictionary with training parameters
    """
    return {
        'learning_rate': params['learning_rate'],
        'weight_decay': params['weight_decay'],
        'batch_size': params['batch_size']
    }
