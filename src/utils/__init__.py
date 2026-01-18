"""Utility functions for data loading and processing."""

from .data_loading import (
    load_and_validate_data,
    load_gene_expression_and_features,
    add_symbolic_regression_features,
    check_data_quality
)

from .hierarchy_detection import (
    detect_hierarchy,
    get_hierarchy_for_classifier,
    visualize_hierarchy,
    compute_celltype_profiles,
    compute_similarity_matrix
)

from .pathway_scores import (
    compute_pathway_scores,
    compute_gene_module_scores,
    add_pathway_features,
    get_pathway_names,
    get_pathway_genes,
    IMMUNE_PATHWAYS
)

from .interpretability import (
    compute_shap_values,
    get_top_features,
    plot_feature_importance,
    plot_shap_summary,
    explain_prediction,
    analyze_class_features,
    print_class_analysis
)

__all__ = [
    'load_and_validate_data',
    'load_gene_expression_and_features',
    'add_symbolic_regression_features',
    'check_data_quality',
    'detect_hierarchy',
    'get_hierarchy_for_classifier',
    'visualize_hierarchy',
    'compute_celltype_profiles',
    'compute_similarity_matrix',
    'compute_pathway_scores',
    'compute_gene_module_scores',
    'add_pathway_features',
    'get_pathway_names',
    'get_pathway_genes',
    'IMMUNE_PATHWAYS',
    'compute_shap_values',
    'get_top_features',
    'plot_feature_importance',
    'plot_shap_summary',
    'explain_prediction',
    'analyze_class_features',
    'print_class_analysis',
]
