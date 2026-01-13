"""
Discover gene regulatory relationships using symbolic regression.

This module uses PySR to find mathematical equations that describe how
marker genes relate to each other within each cell type. For example,
it might discover that CD3D expression = f(CD3E, CD3G) for T cells.

The discovered equations are interpretable and biologically meaningful.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pysr import PySRRegressor
import scanpy as sc


def get_marker_genes_per_celltype(
    adata,
    cell_type_column: str = 'cell_type',
    n_markers: int = 10,
    method: str = 'rank_genes_groups'
) -> Dict[str, List[str]]:
    """
    Identify top marker genes for each cell type.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with cell type labels
    cell_type_column : str
        Column in adata.obs containing cell type labels
    n_markers : int
        Number of top marker genes to identify per cell type
    method : str
        Method to identify markers: 'rank_genes_groups' or 'predefined'
        
    Returns
    -------
    marker_dict : Dict[str, List[str]]
        Dictionary mapping cell type -> list of marker genes
        
    Examples
    --------
    >>> markers = get_marker_genes_per_celltype(adata, n_markers=5)
    >>> print(markers['T cells'])  # ['CD3D', 'CD3E', 'CD3G', 'CD4', 'CD8A']
    """
    
    # Predefined markers based on biological knowledge
    PREDEFINED_MARKERS = {
        'CD4+ T cells': ['CD3D', 'CD3E', 'CD3G', 'CD4', 'IL7R', 'CCR7', 'CD27'],
        'CD8+ T cells': ['CD3D', 'CD3E', 'CD3G', 'CD8A', 'CD8B', 'NKG7', 'GZMA'],
        'CD8+ Cytotoxic T': ['CD8A', 'CD8B', 'GZMA', 'GZMB', 'PRF1', 'NKG7', 'GNLY'],
        'B cells': ['CD19', 'MS4A1', 'CD79A', 'CD79B', 'IGHM', 'IGHD'],
        'NK cells': ['NCAM1', 'NKG7', 'GNLY', 'KLRD1', 'KLRB1', 'GZMA', 'GZMB'],
        'CD14+ Monocytes': ['CD14', 'LYZ', 'S100A8', 'S100A9', 'FCN1', 'VCAN'],
        'CD16+ Monocytes': ['FCGR3A', 'MS4A7', 'LYZ', 'CD14', 'CDKN1C'],
        'Dendritic cells': ['FCER1A', 'CD1C', 'CLEC10A', 'IL3RA'],
        'Megakaryocytes': ['PPBP', 'PF4', 'GP9', 'ITGA2B'],
        'CD34+': ['CD34', 'THY1', 'KIT', 'PROM1']
    }
    
    if method == 'predefined':
        # Use predefined markers, filter for genes present in dataset
        marker_dict = {}
        available_genes = set(adata.var_names)
        
        for cell_type in adata.obs[cell_type_column].unique():
            if cell_type in PREDEFINED_MARKERS:
                # Keep only genes that exist in the dataset
                markers = [g for g in PREDEFINED_MARKERS[cell_type] if g in available_genes]
                if len(markers) >= 3:  # Need at least 3 genes for relationships
                    marker_dict[cell_type] = markers[:n_markers]
            else:
                print(f"Warning: No predefined markers for '{cell_type}', will use rank_genes_groups")
        
        # For cell types without predefined markers, fall back to differential expression
        missing_types = set(adata.obs[cell_type_column].unique()) - set(marker_dict.keys())
        if missing_types and len(marker_dict) > 0:
            print(f"Using differential expression for: {missing_types}")
            method = 'rank_genes_groups'
    
    if method == 'rank_genes_groups' or len(marker_dict) == 0:
        # Use differential expression to find markers
        print("Computing differentially expressed genes...")
        sc.tl.rank_genes_groups(adata, groupby=cell_type_column, method='wilcoxon', n_genes=n_markers)
        
        marker_dict = {}
        for cell_type in adata.obs[cell_type_column].unique():
            # Get top N marker genes for this cell type
            markers = adata.uns['rank_genes_groups']['names'][cell_type][:n_markers].tolist()
            marker_dict[cell_type] = markers
    
    # Print summary
    print(f"\nIdentified marker genes for {len(marker_dict)} cell types:")
    for cell_type, markers in marker_dict.items():
        print(f"  {cell_type}: {len(markers)} markers - {', '.join(markers[:5])}{'...' if len(markers) > 5 else ''}")
    
    return marker_dict


def fit_symbolic_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    target_name: str,
    **pysr_kwargs
) -> PySRRegressor:
    """
    Fit a symbolic regression model to discover mathematical relationships.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features) - gene expression values
    y : np.ndarray
        Target vector (n_samples,) - gene expression to predict
    feature_names : List[str]
        Names of input genes
    target_name : str
        Name of target gene
    **pysr_kwargs
        Additional arguments passed to PySRRegressor
        
    Returns
    -------
    model : PySRRegressor
        Fitted symbolic regression model
        
    Examples
    --------
    >>> # Discover relationship: CD3D = f(CD3E, CD3G)
    >>> X = gene_expr[['CD3E', 'CD3G']].values
    >>> y = gene_expr['CD3D'].values
    >>> model = fit_symbolic_models(X, y, ['CD3E', 'CD3G'], 'CD3D')
    >>> print(model.get_best().equation)  # Shows discovered equation
    """
    
    # Default PySR parameters optimized for gene expression data
    default_params = {
        'niterations': 40,  # Number of iterations (increase for better results)
        'binary_operators': ['+', '-', '*', '/'],
        'unary_operators': ['log', 'exp', 'sqrt', 'square'],
        'populations': 15,  # Number of parallel populations
        'population_size': 33,  # Individuals per population
        'maxsize': 20,  # Maximum equation complexity
        'timeout_in_seconds': 300,  # 5 minute timeout per model
        'parsimony': 0.01,  # Penalty for complexity (smaller = simpler equations)
        'model_selection': 'best',  # or 'accuracy' for most accurate
        'loss': 'loss(prediction, target) = abs(prediction - target)',  # L1 loss
        'random_state': 42,
        'verbosity': 0,  # Reduce console output
        'procs': 4,  # Parallel processes (adjust based on your CPU)
        'multithreading': True,
    }
    
    # Override defaults with user-provided parameters
    params = {**default_params, **pysr_kwargs}
    
    # Create and fit model
    print(f"  Discovering equation for {target_name} using {feature_names}...")
    model = PySRRegressor(**params)
    
    try:
        # Pass feature_names to fit() as variable_names
        model.fit(X, y, variable_names=feature_names)
        
        # Get best equation
        best_eq = model.get_best()
        print(f"    ✓ Found: {target_name} ≈ {best_eq.equation} (loss: {best_eq.loss:.4f})")
        
    except Exception as e:
        print(f"    ✗ Failed to fit model for {target_name}: {e}")
        return None
    
    return model


def discover_gene_relationships(
    adata,
    marker_dict: Dict[str, List[str]],
    cell_type_column: str = 'cell_type',
    use_unsmoothed: bool = True,
    min_cells_per_type: int = 100,
    **pysr_kwargs
) -> Dict[str, Dict[str, PySRRegressor]]:
    """
    Discover gene regulatory relationships for each cell type.
    
    For each cell type, this function:
    1. Filters cells of that type
    2. For each marker gene, tries to predict it from other markers
    3. Uses symbolic regression to find interpretable equations
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    marker_dict : Dict[str, List[str]]
        Mapping of cell type -> list of marker genes
    cell_type_column : str
        Column in adata.obs with cell type labels
    use_unsmoothed : bool
        If True, use adata.layers['unsmoothed_log'] for cleaner relationships
    min_cells_per_type : int
        Minimum number of cells required to fit models
    **pysr_kwargs
        Additional arguments for PySRRegressor
        
    Returns
    -------
    models : Dict[str, Dict[str, PySRRegressor]]
        Nested dict: cell_type -> target_gene -> fitted_model
        
    Examples
    --------
    >>> markers = get_marker_genes_per_celltype(adata)
    >>> models = discover_gene_relationships(adata, markers)
    >>> # Get equation for CD3D in T cells
    >>> t_cell_cd3d_model = models['T cells']['CD3D']
    >>> print(t_cell_cd3d_model.get_best().equation)
    """
    
    print(f"\n{'='*60}")
    print(f"PHASE 1: DISCOVERING GENE REGULATORY RELATIONSHIPS")
    print(f"{'='*60}\n")
    
    # Choose data layer
    if use_unsmoothed and 'unsmoothed_log' in adata.layers:
        print("Using unsmoothed log-transformed data for cleaner relationships")
        expression_data = adata.layers['unsmoothed_log']
    else:
        print("Using adata.X for expression data")
        expression_data = adata.X
    
    # Convert to dense if sparse
    if hasattr(expression_data, 'toarray'):
        expression_data = expression_data.toarray()
    
    # Create DataFrame for easier manipulation
    expr_df = pd.DataFrame(
        expression_data,
        index=adata.obs_names,
        columns=adata.var_names
    )
    expr_df['cell_type'] = adata.obs[cell_type_column].values
    
    # Store models for each cell type and target gene
    all_models = {}
    
    for cell_type, markers in marker_dict.items():
        print(f"\nCell type: {cell_type}")
        print(f"  Markers: {', '.join(markers)}")
        
        # Filter to this cell type
        ct_data = expr_df[expr_df['cell_type'] == cell_type]
        
        if len(ct_data) < min_cells_per_type:
            print(f"  ⚠ Skipping (only {len(ct_data)} cells, need {min_cells_per_type})")
            continue
        
        print(f"  Cells: {len(ct_data)}")
        
        # For each marker gene, try to predict it from the others
        cell_type_models = {}
        
        for i, target_gene in enumerate(markers):
            # Use other markers as predictors
            predictor_genes = [g for g in markers if g != target_gene]
            
            if len(predictor_genes) < 2:
                print(f"  ⚠ Skipping {target_gene} (need at least 2 predictors)")
                continue
            
            # Extract data
            X = ct_data[predictor_genes].values
            y = ct_data[target_gene].values
            
            # Check for variance (can't fit model if target is constant)
            if y.std() < 1e-6:
                print(f"  ⚠ Skipping {target_gene} (no variance)")
                continue
            
            # Fit symbolic regression model
            model = fit_symbolic_models(
                X, y,
                feature_names=predictor_genes,
                target_name=target_gene,
                **pysr_kwargs
            )
            
            if model is not None:
                cell_type_models[target_gene] = model
        
        if len(cell_type_models) > 0:
            all_models[cell_type] = cell_type_models
            print(f"  ✓ Discovered {len(cell_type_models)} relationships")
    
    print(f"\n{'='*60}")
    print(f"DISCOVERY COMPLETE")
    print(f"{'='*60}")
    print(f"Total relationships discovered: {sum(len(v) for v in all_models.values())}")
    print(f"Cell types with models: {len(all_models)}")
    
    return all_models
