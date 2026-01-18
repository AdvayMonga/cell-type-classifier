"""Analyze trained models with SHAP interpretability."""
import argparse
import os
import numpy as np
import torch

from utils import (
    load_and_validate_data,
    load_gene_expression_and_features,
    compute_shap_values,
    get_top_features,
    plot_feature_importance,
    analyze_class_features,
    print_class_analysis
)
from models import CellTypeClassifier


def load_model(model_path, device='cpu'):
    """Load a trained model from checkpoint."""
    data = torch.load(model_path, map_location=device, weights_only=False)

    hidden_sizes = data.get('hidden_sizes', [256, 128])
    use_attention = data.get('use_attention', True)
    attention_hidden = data.get('attention_hidden', 128)

    model = CellTypeClassifier(
        data['input_size'],
        data['num_classes'],
        hidden_sizes=hidden_sizes,
        use_attention=use_attention,
        attention_hidden=attention_hidden
    )
    model.load_state_dict(data['model_state_dict'])
    model.eval()

    return model, data['label_encoder']


def get_feature_names(adata, use_velocity=False):
    """Get feature names from data."""
    names = list(adata.var_names)

    if use_velocity:
        velocity_features = [
            'velocity_pseudotime', 'latent_time', 'velocity_confidence',
            'velocity_magnitude', 'S_score', 'G2M_score',
            'mean_expression', 'expression_variance', 'n_genes_expressed'
        ]
        names.extend([f for f in velocity_features if f in adata.obs.columns])

    return names


def main():
    parser = argparse.ArgumentParser(description='Analyze trained model with SHAP')
    parser.add_argument('--data', type=str, required=True, help='Path to processed .h5ad file')
    parser.add_argument('--model', type=str, default='models/hierarchical_level1.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of samples for SHAP computation')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Number of top features to show')
    parser.add_argument('--save-dir', type=str, default='visualizations',
                        help='Directory to save plots')
    args = parser.parse_args()

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(PROJECT_ROOT, args.data) if not os.path.isabs(args.data) else args.data
    model_path = os.path.join(PROJECT_ROOT, args.model) if not os.path.isabs(args.model) else args.model
    save_dir = os.path.join(PROJECT_ROOT, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("Loading data...")
    adata = load_and_validate_data(data_path)

    print("\nLoading model...")
    model, label_encoder = load_model(model_path)
    class_names = list(label_encoder.classes_)
    print(f"Classes: {class_names}")

    print("\nPreparing features...")
    X = load_gene_expression_and_features(adata, use_velocity=False)
    feature_names = get_feature_names(adata, use_velocity=False)

    if X.shape[1] != model.input_size:
        print(f"Warning: Feature dimension mismatch ({X.shape[1]} vs {model.input_size})")
        feature_names = [f"feature_{i}" for i in range(model.input_size)]

    print(f"\nComputing SHAP values ({args.n_samples} background samples)...")
    print("This may take a few minutes...")

    try:
        shap_values, explainer = compute_shap_values(
            model, X, feature_names,
            n_samples=args.n_samples,
            model_type='nn'
        )

        print("\n" + "="*60)
        print("TOP FEATURES (by mean |SHAP value|)")
        print("="*60)

        top_features = get_top_features(shap_values, feature_names, top_n=args.top_n)
        for i, row in top_features.iterrows():
            print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")

        print("\nGenerating feature importance plot...")
        plot_feature_importance(
            shap_values, feature_names, top_n=args.top_n,
            title='SHAP Feature Importance',
            save_path=os.path.join(save_dir, 'shap_importance.png')
        )

        print("\nAnalyzing class-specific features...")
        class_features = analyze_class_features(shap_values, feature_names, class_names, top_n=10)
        print_class_analysis(class_features)

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Results saved to: {save_dir}/")

    except ImportError:
        print("\nSHAP not installed. Install with: pip install shap")
    except Exception as e:
        print(f"\nSHAP analysis failed: {e}")
        print("Try reducing --n-samples or using a smaller dataset")


if __name__ == '__main__':
    main()
