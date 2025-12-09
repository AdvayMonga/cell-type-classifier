import anndata
import scanpy as sc
import numpy as np
import os
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Select highly variable genes from processed data')
    parser.add_argument('--data', type=str, required=True, help='Path to processed .h5ad file')
    args = parser.parse_args()
    
    # CONFIGURATION - Edit these values to change behavior
    N_GENES = 5000  # Number of highly variable genes to select
    OUTPUT_SUFFIX = f"_{N_GENES//1000}k"  # e.g., "_5k" for 5000 genes
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Handle input path
    if os.path.isabs(args.data):
        input_path = args.data
    else:
        input_path = os.path.join(PROJECT_ROOT, args.data)
    
    # Generate output path automatically
    # e.g., pbmc68k_processed.h5ad -> pbmc68k_processed_5k.h5ad
    base_name = os.path.basename(input_path).replace('.h5ad', '')
    output_name = f"{base_name}{OUTPUT_SUFFIX}.h5ad"
    output_path = os.path.join(os.path.dirname(input_path), output_name)
    
    print(f"\n{'='*70}")
    print(f"GENE SELECTION")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  N_GENES: {N_GENES}")
    print(f"  INPUT: {input_path}")
    print(f"  OUTPUT: {output_path}")
    print(f"{'='*70}")
    
    # Load processed data
    print(f"\nLoading data...")
    adata = anndata.io.read_h5ad(input_path)
    initial_genes = adata.shape[1]
    print(f"Input shape: {adata.shape[0]} cells x {initial_genes} genes")
    
    # Critical markers from Zheng et al. 2017 - FORCE include these
    FORCE_INCLUDE_MARKERS = [
        # T cell markers
        'CD3D', 'CD3E', 'CD3G', 'CD4', 'CD8A', 'CD8B', 'IL7R', 'CCR7',
        # B cell markers  
        'CD19', 'MS4A1', 'CD79A', 'CD79B',
        # NK cell markers
        'NCAM1', 'NKG7', 'GNLY', 'KLRD1',
        # Monocyte markers
        'CD14', 'FCGR3A', 'S100A8', 'S100A9', 'LYZ',
        # DC markers
        'FCER1A', 'CD1C',
        # Stem cell
        'CD34',
        # Pan-leukocyte
        'PTPRC'
    ]
    
    print(f"\n{'='*70}")
    print(f"GENE SELECTION WITH FORCED MARKER INCLUSION")
    print(f"{'='*70}")
    
    # Check which markers exist in the dataset
    existing_markers = [m for m in FORCE_INCLUDE_MARKERS if m in adata.var_names]
    missing_markers = [m for m in FORCE_INCLUDE_MARKERS if m not in adata.var_names]
    
    print(f"  Critical markers from Zheng 2017:")
    print(f"    Found: {len(existing_markers)}/{len(FORCE_INCLUDE_MARKERS)}")
    if missing_markers:
        print(f"    Missing: {', '.join(missing_markers)}")
    
    # Select highly variable genes (don't subset yet)
    print(f"\n  Selecting top {N_GENES} highly variable genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=N_GENES, subset=False)
    
    # FORCE include critical markers
    for marker in existing_markers:
        adata.var.loc[marker, 'highly_variable'] = True
    
    # Now subset to highly variable
    adata = adata[:, adata.var['highly_variable']].copy()
    
    final_genes = adata.shape[1]
    n_forced = len(existing_markers)
    n_selected = final_genes - n_forced
    
    print(f"\n  ✓ Final gene set: {final_genes} genes")
    print(f"    - {n_forced} forced markers (cell surface proteins)")
    print(f"    - {n_selected} highly variable genes")
    print(f"  Genes removed: {initial_genes - final_genes}")
    
    # Verify unsmoothed layer is preserved
    if 'unsmoothed_log' in adata.layers:
        print(f"  ✓ Unsmoothed expression data preserved for classification")
    else:
        print(f"  ⚠ Warning: unsmoothed_log layer not found (using smoothed data)")
    print(f"{'='*70}\n")
    
    # RECOMPUTE n_genes_expressed based on selected genes
    print(f"\nRecomputing n_genes_expressed for {final_genes} selected genes...")
    if hasattr(adata.X, 'toarray'):
        X_dense = adata.X.toarray()
    else:
        X_dense = adata.X
    
    n_genes = np.asarray((X_dense > 0).sum(axis=1)).flatten()
    
    # Normalize to 0-1, then boost by 1.5x (matching velocity_data.py preprocessing)
    nge_min = n_genes.min()
    nge_max = n_genes.max()
    nge_range = nge_max - nge_min
    
    if nge_range < 1e-10:
        normalized = np.full_like(n_genes, 0.5, dtype=float)
        print(f"  ⚠️  All cells have same gene count, setting to 0.75")
    else:
        normalized = (n_genes - nge_min) / nge_range
    
    # Boost by 1.5x to match other velocity features
    velocity_boost_factor = 1.5
    adata.obs['n_genes_expressed'] = normalized * velocity_boost_factor
    
    print(f"  ✓ n_genes_expressed updated:")
    print(f"    Raw range: [{nge_min:.0f}, {nge_max:.0f}]")
    print(f"    Normalized & boosted: [{adata.obs['n_genes_expressed'].min():.2f}, {adata.obs['n_genes_expressed'].max():.2f}]")
    print(f"    Mean: {adata.obs['n_genes_expressed'].mean():.2f}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    print(f"\nSaving to: {output_path}")
    adata.write(output_path)
    
    print(f"\n✓ Gene selection complete!")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
