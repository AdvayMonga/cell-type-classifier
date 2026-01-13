"""
Visualization tools for symbolic regression results.

This module helps interpret and understand the discovered gene relationships.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pysr import PySRRegressor


def print_discovered_equations(
    models: Dict[str, Dict[str, PySRRegressor]],
    top_n: int = 3,
    save_path: Optional[str] = None
) -> None:
    """
    Print all discovered equations in human-readable format.
    
    Parameters
    ----------
    models : Dict[str, Dict[str, PySRRegressor]]
        Discovered models from discover_gene_relationships()
    top_n : int
        Show top N equations per model (by complexity/accuracy trade-off)
    save_path : Optional[str]
        If provided, save equations to this text file
        
    Examples
    --------
    >>> models = discover_gene_relationships(adata, markers)
    >>> print_discovered_equations(models)
    """
    
    print(f"\n{'='*80}")
    print(f"DISCOVERED GENE REGULATORY EQUATIONS")
    print(f"{'='*80}\n")
    
    output_lines = []
    
    for cell_type, ct_models in models.items():
        header = f"\n{'='*80}\n{cell_type.upper()}\n{'='*80}\n"
        print(header)
        output_lines.append(header)
        
        for target_gene, model in ct_models.items():
            # Get equation table (Pareto frontier)
            equations = model.equations_
            
            print(f"\n{target_gene}:")
            output_lines.append(f"\n{target_gene}:")
            
            # Show top N equations
            for i in range(min(top_n, len(equations))):
                eq = equations.iloc[i]
                eq_str = f"  {i+1}. {target_gene} ≈ {eq['equation']}"
                eq_str += f"  (loss: {eq['loss']:.4f}, complexity: {eq['complexity']})"
                print(eq_str)
                output_lines.append(eq_str)
            
            # Show the best equation (what's actually used)
            best_eq = model.get_best()
            best_str = f"\n  → Best (used): {target_gene} = {best_eq.equation}"
            best_str += f"  (loss: {best_eq.loss:.4f})"
            print(best_str)
            output_lines.append(best_str)
    
    # Save to file if requested
    if save_path:
        with open(save_path, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"\n✓ Equations saved to: {save_path}")


def plot_equation_fits(
    adata,
    models: Dict[str, Dict[str, PySRRegressor]],
    cell_type: str,
    target_gene: str,
    save_dir: Optional[str] = None,
    use_unsmoothed: bool = True
) -> None:
    """
    Plot actual vs predicted gene expression for a discovered relationship.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    models : Dict[str, Dict[str, PySRRegressor]]
        Discovered models
    cell_type : str
        Cell type to visualize
    target_gene : str
        Target gene to visualize
    save_dir : Optional[str]
        Directory to save plots
    use_unsmoothed : bool
        Use unsmoothed data
        
    Examples
    --------
    >>> plot_equation_fits(adata, models, 'T cells', 'CD3D', save_dir='visualizations/')
    """
    
    if cell_type not in models or target_gene not in models[cell_type]:
        print(f"No model found for {cell_type} -> {target_gene}")
        return
    
    model = models[cell_type][target_gene]
    
    # Get expression data
    if use_unsmoothed and 'unsmoothed_log' in adata.layers:
        expression_data = adata.layers['unsmoothed_log']
    else:
        expression_data = adata.X
    
    if hasattr(expression_data, 'toarray'):
        expression_data = expression_data.toarray()
    
    expr_df = pd.DataFrame(expression_data, columns=adata.var_names)
    expr_df['cell_type'] = adata.obs['cell_type'].values
    
    # Filter to cell type
    ct_data = expr_df[expr_df['cell_type'] == cell_type]
    
    # Get predictions - try different attribute names for compatibility
    if hasattr(model, 'feature_names_in_'):
        predictor_genes = model.feature_names_in_
    elif hasattr(model, 'variable_names'):
        predictor_genes = model.variable_names
    else:
        raise AttributeError("Cannot find feature names in model")
    
    X = ct_data[predictor_genes].values
    y_actual = ct_data[target_gene].values
    y_pred = model.predict(X)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Actual vs Predicted scatter
    ax1 = axes[0]
    ax1.scatter(y_actual, y_pred, alpha=0.3, s=10)
    
    # Add diagonal line (perfect prediction)
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect fit')
    
    # Calculate R²
    r_squared = 1 - np.sum((y_actual - y_pred)**2) / np.sum((y_actual - y_actual.mean())**2)
    
    ax1.set_xlabel(f'Actual {target_gene} expression', fontsize=12)
    ax1.set_ylabel(f'Predicted {target_gene} expression', fontsize=12)
    ax1.set_title(f'{cell_type}: {target_gene} Prediction\nR² = {r_squared:.3f}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    ax2 = axes[1]
    residuals = y_actual - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.3, s=10)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel(f'Predicted {target_gene} expression', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add equation to figure
    best_eq = model.get_best()
    fig.suptitle(f'Equation: {target_gene} = {best_eq.equation}', 
                 fontsize=12, y=1.02, fontweight='bold')
    
    plt.tight_layout()
    
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{cell_type.replace(' ', '_')}_{target_gene}_fit.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {filename}")
    
    plt.show()


def visualize_regulatory_network(
    models: Dict[str, Dict[str, PySRRegressor]],
    cell_type: str,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize gene regulatory network as a graph.
    
    Shows which genes predict which other genes in a cell type.
    
    Parameters
    ----------
    models : Dict[str, Dict[str, PySRRegressor]]
        Discovered models
    cell_type : str
        Cell type to visualize
    save_path : Optional[str]
        Path to save network visualization
        
    Examples
    --------
    >>> visualize_regulatory_network(models, 'T cells', save_path='t_cell_network.png')
    """
    
    try:
        import networkx as nx
    except ImportError:
        print("networkx not installed. Install with: pip install networkx")
        return
    
    if cell_type not in models:
        print(f"No models found for {cell_type}")
        return
    
    # Create directed graph
    G = nx.DiGraph()
    
    ct_models = models[cell_type]
    
    # Add edges: predictor genes -> target gene
    for target_gene, model in ct_models.items():
        # Get feature names - try different attribute names for compatibility
        if hasattr(model, 'feature_names_in_'):
            predictor_genes = model.feature_names_in_
        elif hasattr(model, 'variable_names'):
            predictor_genes = model.variable_names
        else:
            continue  # Skip if can't find feature names
        
        # Add nodes
        G.add_node(target_gene)
        for pred_gene in predictor_genes:
            G.add_node(pred_gene)
            
            # Add edge with weight = prediction quality
            best_eq = model.get_best()
            weight = 1.0 / (1.0 + best_eq.loss)  # Lower loss = higher weight
            G.add_edge(pred_gene, target_gene, weight=weight)
    
    # Draw network
    plt.figure(figsize=(12, 10))
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', 
                           alpha=0.9, edgecolors='black', linewidths=2)
    
    # Draw edges with varying thickness based on weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights], 
                           alpha=0.6, edge_color='gray', 
                           arrows=True, arrowsize=20, arrowstyle='->')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title(f'Gene Regulatory Network: {cell_type}', fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved network visualization: {save_path}")
    
    plt.show()


def plot_regulatory_feature_importance(
    feature_importances: Dict[str, float],
    top_n: int = 20,
    save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance showing which regulatory features matter most.
    
    Parameters
    ----------
    feature_importances : Dict[str, float]
        Feature name -> importance score
    top_n : int
        Show top N features
    save_path : Optional[str]
        Path to save plot
        
    Examples
    --------
    >>> # After training classifier with regulatory features
    >>> importances = get_feature_importances(model)
    >>> plot_regulatory_feature_importance(importances)
    """
    
    # Sort by importance
    sorted_features = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Get top N regulatory features
    regulatory_features = [(k, v) for k, v in sorted_features if 'reg_score' in k][:top_n]
    
    if len(regulatory_features) == 0:
        print("No regulatory features found in importance scores")
        return
    
    names, values = zip(*regulatory_features)
    
    # Plot
    plt.figure(figsize=(10, max(6, len(names) * 0.3)))
    colors = ['green' if v > 0 else 'red' for v in values]
    plt.barh(range(len(names)), values, color=colors, alpha=0.7)
    plt.yticks(range(len(names)), names, fontsize=9)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Top Regulatory Feature Importances', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved importance plot: {save_path}")
    
    plt.show()
