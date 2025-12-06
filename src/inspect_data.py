import scvelo as scv
import os
import numpy as np
import anndata
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Inspect PBMC dataset')
    parser.add_argument('--data', type=str, default='data/PBMC/pbmc68k.h5ad',
                        help='Path to the dataset file to inspect')
    args = parser.parse_args()

    # Dataset configuration
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Use provided path or default
    if os.path.isabs(args.data):
        DATA_PATH = args.data
    else:
        DATA_PATH = os.path.join(PROJECT_ROOT, args.data)

    # Load dataset
    if not os.path.exists(DATA_PATH):
        print(f"\nDataset not found at: {DATA_PATH}")
        print("Downloading raw PBMC68k dataset...")
        adata = scv.datasets.pbmc68k()
        RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data/PBMC/pbmc68k.h5ad")
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        adata.write(RAW_DATA_PATH)
        print(f"Saved to: {RAW_DATA_PATH}")
    else:
        print(f"Loading dataset from: {DATA_PATH}")
        adata = anndata.io.read_h5ad(DATA_PATH)

    print(f"\n{'='*80}")
    print(f"PBMC68K DATASET INSPECTION")
    print(f"{'='*80}")
    
    # Basic info
    print(f"\nDataset Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"Data type: {type(adata.X)}")
    print(f"Layers: {list(adata.layers.keys())}")
    
    # Gene names
    print(f"\n{'='*80}")
    print(f"GENE NAMES")
    print(f"{'='*80}")
    print(f"\nFirst 50 genes:")
    for i, gene in enumerate(adata.var_names[:50]):
        print(f"  {i+1:3d}. {gene}")
    
    print(f"\nLast 20 genes:")
    for i, gene in enumerate(adata.var_names[-20:]):
        print(f"  {adata.shape[1]-20+i+1:3d}. {gene}")
    
    print(f"\nRandom sample of 30 genes:")
    import random
    random.seed(42)
    sample_indices = sorted(random.sample(range(len(adata.var_names)), 30))
    for idx in sample_indices:
        print(f"  {idx+1:5d}. {adata.var_names[idx]}")
    
    # Gene name patterns
    print(f"\n{'='*80}")
    print(f"GENE NAME PATTERNS")
    print(f"{'='*80}")
    gene_names_upper = adata.var_names.str.upper()
    
    print(f"\nPrefix patterns:")
    print(f"  MT- (mitochondrial): {adata.var_names.str.startswith('MT-').sum()}")
    print(f"  RP (ribosomal/pseudogenes): {adata.var_names.str.startswith('RP').sum()}")
    print(f"  RPS (ribosomal protein small): {adata.var_names.str.startswith('RPS').sum()}")
    print(f"  RPL (ribosomal protein large): {adata.var_names.str.startswith('RPL').sum()}")
    print(f"  OR (olfactory receptors): {adata.var_names.str.startswith('OR').sum()}")
    print(f"  LINC (long intergenic non-coding): {adata.var_names.str.startswith('LINC').sum()}")
    
    print(f"\nCase patterns:")
    print(f"  All uppercase: {adata.var_names.str.isupper().sum()}")
    print(f"  Contains lowercase: {adata.var_names.str.contains('[a-z]').sum()}")
    print(f"  Contains numbers: {adata.var_names.str.contains('[0-9]').sum()}")
    print(f"  Contains hyphens: {adata.var_names.str.contains('-').sum()}")
    
    # Search for common marker genes
    print(f"\n{'='*80}")
    print(f"COMMON MARKER GENES")
    print(f"{'='*80}")
    
    immune_markers = ['CD3D', 'CD3E', 'CD4', 'CD8A', 'CD8B', 'MS4A1', 'CD19', 'CD14', 
                      'FCGR3A', 'CD68', 'NCAM1', 'IL7R', 'CCR7', 'NKG7', 'GNLY']
    
    print(f"\nImmune cell markers:")
    found_markers = []
    for marker in immune_markers:
        if marker.upper() in gene_names_upper.values:
            idx = list(gene_names_upper.values).index(marker.upper())
            actual_name = adata.var_names[idx]
            print(f"  ✓ {marker:10s} -> {actual_name}")
            found_markers.append(actual_name)
        else:
            print(f"  ✗ {marker:10s} -> Not found")
    
    # Search for cell cycle genes
    print(f"\nCell cycle genes (Regev lab list):")

    s_genes = ['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6', 
               'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'HELLS', 'RFC2', 'RPA2', 'NASP', 
               'RAD51AP1', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 
               'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 
               'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8']

    g2m_genes = ['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 
                 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'SMC4', 'CCNB2', 
                 'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 
                 'KIF20B', 'HJURP', 'CDCA3', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 
                 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1', 
                 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA']
    
    cell_cycle_test = s_genes + g2m_genes

    found_cc = []
    for gene in cell_cycle_test:
        if gene.upper() in gene_names_upper.values:
            idx = list(gene_names_upper.values).index(gene.upper())
            actual_name = adata.var_names[idx]
            print(f"  ✓ {gene:10s} -> {actual_name}")
            found_cc.append(actual_name)
        else:
            print(f"  ✗ {gene:10s} -> Not found")

    print(f"\n✓ Found {len(found_cc)}/{len(cell_cycle_test)} test cell cycle genes")

    # Expression statistics
    print(f"\n{'='*80}")
    print(f"EXPRESSION STATISTICS")
    print(f"{'='*80}")
    
    if hasattr(adata.X, 'toarray'):
        X_check = adata.X.toarray()
    else:
        X_check = np.array(adata.X)
    
    print(f"\nData value range:")
    print(f"  Min: {X_check.min():.4f}")
    print(f"  Max: {X_check.max():.4f}")
    print(f"  Mean: {X_check.mean():.4f}")
    print(f"  Median: {np.median(X_check):.4f}")
    print(f"  Std: {X_check.std():.4f}")
    
    # Sparsity
    sparsity = (X_check == 0).sum() / X_check.size * 100
    print(f"\nSparsity: {sparsity:.2f}% zeros")
    
    # Cells per gene
    cells_per_gene = (X_check > 0).sum(axis=0)
    if len(cells_per_gene.shape) > 1:
        cells_per_gene = np.asarray(cells_per_gene).flatten()
    
    print(f"\nGenes by expression frequency:")
    print(f"  Genes in ≥1 cells: {(cells_per_gene >= 1).sum()}")
    print(f"  Genes in ≥10 cells: {(cells_per_gene >= 10).sum()}")
    print(f"  Genes in ≥100 cells: {(cells_per_gene >= 100).sum()}")
    print(f"  Genes in ≥1000 cells: {(cells_per_gene >= 1000).sum()}")
    print(f"  Genes in ≥10000 cells: {(cells_per_gene >= 10000).sum()}")
    
    # Observation (cell) metadata
    print(f"\n{'='*80}")
    print(f"CELL METADATA")
    print(f"{'='*80}")
    print(f"\nColumns in adata.obs:")
    for col in adata.obs.columns:
        print(f"  - {col}: {adata.obs[col].dtype}")
        if adata.obs[col].dtype == 'object' or adata.obs[col].dtype.name == 'category':
            n_unique = adata.obs[col].nunique()
            print(f"    Unique values: {n_unique}")
            if n_unique <= 20:
                print(f"    Values: {list(adata.obs[col].unique())}")
    
    print(f"\n{'='*80}")

if __name__ == '__main__':
    main()
