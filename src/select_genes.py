import anndata
import scanpy as sc
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
    
    # Select highly variable genes
    print(f"\nSelecting top {N_GENES} highly variable genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=N_GENES, subset=True)
    
    final_genes = adata.shape[1]
    print(f"Output shape: {adata.shape[0]} cells x {final_genes} genes")
    print(f"Genes removed: {initial_genes - final_genes}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    print(f"\nSaving to: {output_path}")
    adata.write(output_path)
    
    print(f"\n✓ Gene selection complete!")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
