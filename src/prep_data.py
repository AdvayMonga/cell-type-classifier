"""Preprocessing pipeline for PBMC data with RNA velocity."""
import scvelo as scv
import scanpy as sc
import numpy as np
import os
import argparse


def get_protected_genes():
    """Return gene lists that should never be filtered (cell cycle, markers)."""
    s_genes = [
        'MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6',
        'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'HELLS', 'RFC2', 'RPA2', 'NASP',
        'RAD51AP1', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2',
        'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2',
        'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8'
    ]

    g2m_genes = [
        'HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2',
        'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'SMC4', 'CCNB2',
        'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1',
        'KIF20B', 'HJURP', 'CDCA3', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1',
        'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1',
        'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA'
    ]

    cell_surface_markers = [
        'CD3D', 'CD3E', 'CD3G', 'CD4', 'CD8A', 'CD8B', 'IL7R', 'CCR7', 'CD27', 'SELL',
        'CD19', 'MS4A1', 'CD79A', 'CD79B',
        'NCAM1', 'NKG7', 'GNLY', 'KLRD1', 'KLRB1',
        'CD14', 'FCGR3A', 'S100A8', 'S100A9', 'LYZ', 'CD68',
        'FCER1A', 'CD1C', 'CD34', 'PTPRC',
    ]

    return {'s_phase': s_genes, 'g2m_phase': g2m_genes, 'markers': cell_surface_markers}


def filter_low_quality_genes(adata, min_cells=10, remove_mt=True, remove_ribo=True,
                              remove_pseudogenes=True, remove_olfactory=True):
    """Remove low-quality genes while protecting critical markers."""
    print("\nRemoving low-quality genes...")

    protected = get_protected_genes()
    all_protected = protected['s_phase'] + protected['g2m_phase'] + protected['markers']
    protected_upper = set([g.upper() for g in all_protected])

    gene_names_upper = adata.var_names.str.upper()
    is_protected = gene_names_upper.isin(protected_upper)
    genes_to_remove = np.zeros(len(gene_names_upper), dtype=bool)

    if remove_pseudogenes:
        pseudogenes = gene_names_upper.str.match(r'^RP[^SL]')
        genes_to_remove |= pseudogenes
        print(f"   Pseudogenes: {pseudogenes.sum()}")

    if remove_olfactory:
        olfactory = gene_names_upper.str.startswith('OR')
        genes_to_remove |= olfactory
        print(f"   Olfactory: {olfactory.sum()}")

    if remove_mt:
        mitochondrial = gene_names_upper.str.startswith('MT-')
        genes_to_remove |= mitochondrial
        print(f"   Mitochondrial: {mitochondrial.sum()}")

    if remove_ribo:
        ribosomal = gene_names_upper.str.startswith(('RPS', 'RPL'))
        genes_to_remove |= ribosomal
        print(f"   Ribosomal: {ribosomal.sum()}")

    genes_to_remove = genes_to_remove & ~is_protected
    print(f"   Protected: {is_protected.sum()}, To remove: {genes_to_remove.sum()}")

    adata_filtered = adata[:, ~genes_to_remove].copy()
    print(f"   After filtering: {adata_filtered.shape[0]} cells x {adata_filtered.shape[1]} genes")

    return adata_filtered, protected


def compute_velocity_features(adata, n_pcs=30, n_neighbors=30):
    """Compute RNA velocity on preprocessed data."""
    print("\n   Computing PCA, neighbors, and moments...")

    adata.layers['unsmoothed_log'] = adata.X.copy()
    print(f"   Saved unsmoothed data to 'unsmoothed_log' layer")

    sc.tl.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

    print("\nEstimating RNA velocity...")
    scv.tl.velocity(adata, mode='stochastic')
    scv.tl.velocity_graph(adata, n_jobs=1)

    print("\nComputing UMAP...")
    sc.tl.umap(adata)

    return adata


def extract_velocity_features(adata, protected_genes):
    """Extract velocity-based features for classification."""
    print("\nExtracting velocity features...")

    velocity_matrix = adata.layers['velocity']
    if hasattr(velocity_matrix, 'toarray'):
        velocity_matrix = velocity_matrix.toarray()

    velocity_magnitude = np.sqrt((velocity_matrix ** 2).sum(axis=1))

    adata.obs['velocity_pseudotime'] = (velocity_magnitude - velocity_magnitude.min()) / (velocity_magnitude.max() - velocity_magnitude.min())
    print(f"   Added velocity_pseudotime")

    velocity_direction = np.sign(velocity_matrix).mean(axis=1)
    if hasattr(velocity_direction, 'A1'):
        velocity_direction = velocity_direction.A1
    else:
        velocity_direction = np.asarray(velocity_direction).flatten()

    vd_min, vd_max = velocity_direction.min(), velocity_direction.max()
    if vd_max - vd_min < 1e-10:
        adata.obs['latent_time'] = np.full(len(velocity_direction), 0.5)
    else:
        adata.obs['latent_time'] = (velocity_direction - vd_min) / (vd_max - vd_min)
    print(f"   Added latent_time")

    scv.tl.velocity_confidence(adata)
    print(f"   Added velocity_confidence")

    adata.obs['velocity_magnitude'] = velocity_magnitude
    print(f"   Added velocity_magnitude")

    adata = compute_cell_cycle_scores(adata, protected_genes)
    adata = compute_expression_statistics(adata)

    return adata


def compute_cell_cycle_scores(adata, protected_genes):
    """Compute cell cycle scores using protected gene lists."""
    print("\nComputing cell cycle scores...")

    s_genes = protected_genes['s_phase']
    g2m_genes = protected_genes['g2m_phase']

    try:
        s_genes_present = [g for g in s_genes if g.upper() in adata.var_names.str.upper().values]
        g2m_genes_present = [g for g in g2m_genes if g.upper() in adata.var_names.str.upper().values]
        print(f"   S-phase: {len(s_genes_present)}/{len(s_genes)}, G2M: {len(g2m_genes_present)}/{len(g2m_genes)}")

        if len(s_genes_present) >= 5 and len(g2m_genes_present) >= 5:
            scv.tl.score_genes_cell_cycle(adata)
            print(f"   Added S_score and G2M_score")
        else:
            raise ValueError("Not enough cell cycle genes")

    except Exception as e:
        print(f"   Using expression stats as proxy: {e}")
        if hasattr(adata.X, 'toarray'):
            X_dense = adata.X.toarray()
        else:
            X_dense = adata.X

        mean_expr = X_dense.mean(axis=1)
        expr_var = X_dense.var(axis=1)

        adata.obs['S_score'] = (mean_expr - mean_expr.min()) / (mean_expr.max() - mean_expr.min())
        adata.obs['G2M_score'] = (expr_var - expr_var.min()) / (expr_var.max() - expr_var.min())

    return adata


def compute_expression_statistics(adata):
    """Compute per-cell expression statistics."""
    print("\nComputing expression statistics...")

    if hasattr(adata.X, 'toarray'):
        X_dense = adata.X.toarray()
    else:
        X_dense = adata.X

    adata.obs['mean_expression'] = np.asarray(X_dense.mean(axis=1)).flatten()
    adata.obs['expression_variance'] = np.asarray(X_dense.var(axis=1)).flatten()
    adata.obs['n_genes_expressed'] = np.asarray((X_dense > 0).sum(axis=1)).flatten()

    print(f"   Added mean_expression, expression_variance, n_genes_expressed")
    return adata


def normalize_and_boost_features(adata, boost_factor=1.5):
    """Normalize velocity features to 0-1 range and boost."""
    print("\nNormalizing and boosting velocity features...")

    velocity_features = [
        'velocity_pseudotime', 'latent_time', 'velocity_confidence',
        'velocity_magnitude', 'S_score', 'G2M_score',
        'mean_expression', 'expression_variance', 'n_genes_expressed'
    ]

    for feat in velocity_features:
        if feat not in adata.obs.columns:
            continue

        values = adata.obs[feat].values
        values = np.nan_to_num(values, nan=0.0)

        val_min, val_max = values.min(), values.max()
        if val_max - val_min < 1e-10:
            normalized = np.full_like(values, 0.5)
        else:
            normalized = (values - val_min) / (val_max - val_min)

        adata.obs[feat] = normalized * boost_factor
        print(f"   {feat}: [{adata.obs[feat].min():.4f}, {adata.obs[feat].max():.4f}]")

    return adata


def main():
    parser = argparse.ArgumentParser(description='Preprocess PBMC data with scVelo')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    args = parser.parse_args()

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data/PBMC/pbmc68k.h5ad")
    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data/PBMC/pbmc68k_processed.h5ad")
    VIZ_DIR = os.path.join(PROJECT_ROOT, 'visualizations')

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    if not os.path.exists(RAW_DATA_PATH):
        print("\nDownloading dataset...")
        adata = scv.datasets.pbmc68k()
        adata.write(RAW_DATA_PATH)
    else:
        print("Loading existing dataset...")
        import anndata
        adata = anndata.io.read_h5ad(RAW_DATA_PATH)

    scv.set_figure_params('scvelo')

    print(f"\n{'='*60}")
    print(f"INITIAL: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"{'='*60}")

    adata, protected = filter_low_quality_genes(adata)

    print("\nPreprocessing...")
    scv.pp.filter_genes(adata, min_cells=10)
    scv.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    print(f"   Final: {adata.n_vars} genes")

    adata = compute_velocity_features(adata)
    adata = extract_velocity_features(adata, protected)
    adata = normalize_and_boost_features(adata)

    if args.plot:
        print("\nGenerating visualizations...")
        import matplotlib.pyplot as plt

        scv.pl.velocity_embedding_stream(adata, basis='umap')
        plt.savefig(os.path.join(VIZ_DIR, 'velocity_stream.png'), dpi=300)
        plt.close()

    print("\nAssigning cell types...")
    if 'celltype' in adata.obs.columns:
        adata.obs['cell_type'] = adata.obs['celltype']
    elif 'clusters' in adata.obs.columns:
        adata.obs['cell_type'] = adata.obs['clusters']
    else:
        sc.tl.leiden(adata, flavor='igraph', resolution=0.5)
        adata.obs['cell_type'] = adata.obs['leiden']

    print("\nSaving...")
    adata.write(PROCESSED_DATA_PATH)

    print(f"\n{'='*60}")
    print(f"COMPLETE: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"Cell types: {adata.obs['cell_type'].nunique()}")
    print(f"Saved to: {PROCESSED_DATA_PATH}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
