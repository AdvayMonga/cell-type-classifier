"""SHAP-based model interpretability utilities."""
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt


def compute_shap_values(model, X: np.ndarray, feature_names: Optional[List[str]] = None,
                        n_samples: int = 100, model_type: str = 'nn') -> Tuple[np.ndarray, any]:
    """
    Compute SHAP values for model predictions.

    Args:
        model: Trained model (PyTorch nn.Module or sklearn-style)
        X: Feature matrix (n_cells, n_features)
        feature_names: Names for features
        n_samples: Number of background samples for SHAP
        model_type: 'nn' for neural networks, 'tree' for XGBoost/RF

    Returns:
        Tuple of (shap_values, explainer)
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP not installed. Run: pip install shap")

    np.random.seed(42)
    bg_indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    background = X[bg_indices]

    if model_type == 'nn':
        import torch
        model.eval()

        def predict_fn(x):
            with torch.no_grad():
                tensor = torch.FloatTensor(x)
                output = model(tensor)
                probs = torch.softmax(output, dim=1)
                return probs.numpy()

        explainer = shap.KernelExplainer(predict_fn, background)
        sample_indices = np.random.choice(len(X), min(100, len(X)), replace=False)
        shap_values = explainer.shap_values(X[sample_indices])

    elif model_type == 'tree':
        explainer = shap.TreeExplainer(model, background)
        shap_values = explainer.shap_values(X)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return shap_values, explainer


def get_top_features(shap_values: np.ndarray, feature_names: List[str],
                      top_n: int = 20, class_idx: Optional[int] = None) -> pd.DataFrame:
    """
    Get top features by mean absolute SHAP value.

    Args:
        shap_values: SHAP values (list of arrays for multi-class)
        feature_names: Names of features
        top_n: Number of top features to return
        class_idx: Specific class index (None = aggregate across classes)

    Returns:
        DataFrame with feature names and importance scores
    """
    if isinstance(shap_values, list):
        if class_idx is not None:
            values = np.abs(shap_values[class_idx]).mean(axis=0)
        else:
            values = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        values = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': values
    })
    importance_df = importance_df.sort_values('importance', ascending=False)

    return importance_df.head(top_n)


def plot_feature_importance(shap_values: np.ndarray, feature_names: List[str],
                             top_n: int = 20, title: str = 'Feature Importance',
                             save_path: Optional[str] = None) -> None:
    """
    Plot feature importance from SHAP values.

    Args:
        shap_values: SHAP values
        feature_names: Names of features
        top_n: Number of top features to show
        title: Plot title
        save_path: Path to save figure (optional)
    """
    top_features = get_top_features(shap_values, feature_names, top_n)

    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    plt.barh(range(len(top_features)), top_features['importance'].values[::-1], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'].values[::-1])
    plt.xlabel('Mean |SHAP value|')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_shap_summary(shap_values: np.ndarray, X: np.ndarray, feature_names: List[str],
                       class_idx: int = 0, top_n: int = 20,
                       save_path: Optional[str] = None) -> None:
    """
    Plot SHAP summary (beeswarm) showing feature effects.

    Args:
        shap_values: SHAP values
        X: Feature matrix
        feature_names: Names of features
        class_idx: Class index to visualize
        top_n: Number of top features to show
        save_path: Path to save figure
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP not installed. Run: pip install shap")

    if isinstance(shap_values, list):
        sv = shap_values[class_idx]
    else:
        sv = shap_values

    top_features = get_top_features([sv] if not isinstance(shap_values, list) else shap_values,
                                     feature_names, top_n, class_idx)
    top_indices = [feature_names.index(f) for f in top_features['feature'].values]

    plt.figure(figsize=(10, max(6, top_n * 0.4)))
    shap.summary_plot(sv[:, top_indices], X[:, top_indices],
                      feature_names=[feature_names[i] for i in top_indices],
                      show=False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def explain_prediction(model, x: np.ndarray, shap_values: np.ndarray,
                        feature_names: List[str], class_names: List[str],
                        top_n: int = 10) -> dict:
    """
    Explain a single prediction.

    Args:
        model: Trained model
        x: Single sample (1D array)
        shap_values: SHAP values for this sample
        feature_names: Names of features
        class_names: Names of classes
        top_n: Number of top contributing features

    Returns:
        Dict with prediction details and explanations
    """
    import torch

    model.eval()
    with torch.no_grad():
        tensor = torch.FloatTensor(x.reshape(1, -1))
        output = model(tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]
        pred_class = probs.argmax()

    if isinstance(shap_values, list):
        sv = shap_values[pred_class]
    else:
        sv = shap_values

    if len(sv.shape) > 1:
        sv = sv[0]

    sorted_idx = np.argsort(np.abs(sv))[::-1][:top_n]

    contributions = []
    for idx in sorted_idx:
        contributions.append({
            'feature': feature_names[idx],
            'value': x[idx],
            'shap_value': sv[idx],
            'direction': 'positive' if sv[idx] > 0 else 'negative'
        })

    return {
        'predicted_class': class_names[pred_class],
        'predicted_probability': probs[pred_class],
        'all_probabilities': {class_names[i]: probs[i] for i in range(len(class_names))},
        'top_contributions': contributions
    }


def analyze_class_features(shap_values: np.ndarray, feature_names: List[str],
                            class_names: List[str], top_n: int = 10) -> dict:
    """
    Analyze which features are most important for each class.

    Args:
        shap_values: List of SHAP value arrays (one per class)
        feature_names: Names of features
        class_names: Names of classes
        top_n: Number of top features per class

    Returns:
        Dict mapping class names to their top features
    """
    if not isinstance(shap_values, list):
        return {'all_classes': get_top_features(shap_values, feature_names, top_n)}

    class_features = {}
    for i, class_name in enumerate(class_names):
        sv = shap_values[i]
        mean_abs = np.abs(sv).mean(axis=0)

        top_indices = np.argsort(mean_abs)[::-1][:top_n]
        class_features[class_name] = [
            {'feature': feature_names[idx], 'importance': mean_abs[idx]}
            for idx in top_indices
        ]

    return class_features


def print_class_analysis(class_features: dict) -> None:
    """Print class-specific feature importance analysis."""
    print("\n" + "="*60)
    print("CLASS-SPECIFIC FEATURE IMPORTANCE")
    print("="*60)

    for class_name, features in class_features.items():
        print(f"\n{class_name}:")
        for i, feat in enumerate(features[:5], 1):
            print(f"  {i}. {feat['feature']}: {feat['importance']:.4f}")
