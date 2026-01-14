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
]
