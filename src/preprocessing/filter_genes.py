"""Gene filtering with protected gene support."""
import numpy as np


def get_protected_genes():
    """
    Return lists of genes that should never be filtered.
    
    Returns:
        dict: Dictionary with keys 's_phase', 'g2m_phase', 'markers' containing gene lists
    """
    # S-phase cell cycle genes to ALWAYS keep
    s_genes = [
        'MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6', 
        'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'HELLS', 'RFC2', 'RPA2', 'NASP', 
        'RAD51AP1', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 
        'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 
        'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8'
    ]

    # G2M-phase cell cycle genes to ALWAYS keep
    g2m_genes = [
        'HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 
        'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'SMC4', 'CCNB2', 
        'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 
        'KIF20B', 'HJURP', 'CDCA3', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 
        'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1', 
        'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA'
    ]

    # CRITICAL: Cell surface markers from Zheng et al. 2017 Nature Communications
    # These are the markers used for FACS sorting and are essential for classification
    cell_surface_markers = [
        # T cell markers (CD3 complex)
        'CD3D', 'CD3E', 'CD3G',          # Pan T cell markers
        'CD4',                            # T helper cells
        'CD8A', 'CD8B',                  # Cytotoxic T cells
        'IL7R',                           # CD127 - Naive T cells
        'CCR7',                           # Naive T cell homing
        'CD27',                           # Memory T cells
        'SELL',                           # L-selectin (CD62L) - Naive
        
        # B cell markers
        'CD19',                           # Pan B cell
        'MS4A1',                          # CD20
        'CD79A', 'CD79B',                # B cell receptor components
        
        # NK cell markers
        'NCAM1',                          # CD56 - NK cells
        'NKG7', 'GNLY',                  # NK cytotoxicity
        'KLRD1', 'KLRB1',                # Killer cell receptors
        
        # Monocyte markers
        'CD14',                           # Classical monocytes
        'FCGR3A',                         # CD16 - Non-classical monocytes
        'S100A8', 'S100A9',              # Monocyte-specific calgranulins
        'LYZ',                            # Lysozyme
        'CD68',                           # Macrophage marker
        
        # Dendritic cell markers
        'FCER1A',                         # Classical DC
        'CD1C',                           # cDC2
        
        # Stem cell markers
        'CD34',                           # Hematopoietic stem cells
        
        # Pan-leukocyte
        'PTPRC',                          # CD45 - All leukocytes
    ]
    
    return {
        's_phase': s_genes,
        'g2m_phase': g2m_genes,
        'markers': cell_surface_markers
    }


def filter_low_quality_genes(adata, min_cells=10, remove_mt=True, remove_ribo=True, 
                              remove_pseudogenes=True, remove_olfactory=True):
    """
    Remove low-quality genes while protecting critical markers.
    
    Args:
        adata: AnnData object
        min_cells: Minimum number of cells expressing a gene
        remove_mt: Remove mitochondrial genes (MT-*)
        remove_ribo: Remove ribosomal genes (RPS*, RPL*)
        remove_pseudogenes: Remove pseudogenes (RP* but not RPS/RPL)
        remove_olfactory: Remove olfactory receptors (OR*)
    
    Returns:
        tuple: (filtered_adata, protected_genes_dict)
    """
    print("\nRemoving low-quality genes...")
    
    protected = get_protected_genes()
    all_protected = protected['s_phase'] + protected['g2m_phase'] + protected['markers']
    protected_upper = set([g.upper() for g in all_protected])
    
    print(f"   Protected gene categories:")
    print(f"     - S-phase cell cycle: {len(protected['s_phase'])} genes")
    print(f"     - G2M-phase cell cycle: {len(protected['g2m_phase'])} genes")
    print(f"     - Cell surface markers (Zheng 2017): {len(protected['markers'])} genes")
    
    # Identify genes to remove
    gene_names_upper = adata.var_names.str.upper()
    is_protected = gene_names_upper.isin(protected_upper)
    
    genes_to_remove = np.zeros(len(gene_names_upper), dtype=bool)
    
    if remove_pseudogenes:
        pseudogenes = gene_names_upper.str.match(r'^RP[^SL]')  # RP but not RPS or RPL
        genes_to_remove |= pseudogenes
        print(f"   Pseudogenes (RP*): {pseudogenes.sum()}")
    
    if remove_olfactory:
        olfactory = gene_names_upper.str.startswith('OR')
        genes_to_remove |= olfactory
        print(f"   Olfactory receptors (OR*): {olfactory.sum()}")
    
    if remove_mt:
        mitochondrial = gene_names_upper.str.startswith('MT-')
        genes_to_remove |= mitochondrial
        print(f"   Mitochondrial genes (MT-*): {mitochondrial.sum()}")
    
    if remove_ribo:
        ribosomal = gene_names_upper.str.startswith(('RPS', 'RPL'))
        genes_to_remove |= ribosomal
        print(f"   Ribosomal genes (RPS*, RPL*): {ribosomal.sum()}")
    
    # NEVER remove protected genes
    genes_to_remove = genes_to_remove & ~is_protected
    
    print(f"   Protected genes found: {is_protected.sum()}")
    print(f"   Total genes to remove: {genes_to_remove.sum()}")
    
    adata_filtered = adata[:, ~genes_to_remove].copy()
    print(f"   ✓ After filtering: {adata_filtered.shape[0]} cells x {adata_filtered.shape[1]} genes")
    
    return adata_filtered, protected
