import scanpy as sc
from .config import PROCESSED_DATA_DIR, RAW_DATA_DIR, V1_CODES, V2_CODES
from sklearn.model_selection import StratifiedGroupKFold
from typing import List, Tuple

def keep_new_donors(filename: str) -> sc.AnnData:

    raw_path = RAW_DATA_DIR / filename
    adata = sc.read_h5ad(raw_path)
    mask = adata.obs['donor'].isin(V2_CODES) & ~adata.obs['donor'].isin(V1_CODES)
    return adata[mask].copy()

def holdout_TSP30(adata: sc.AnnData) -> tuple[sc.AnnData, sc.AnnData]:

    is_test = adata.obs['donor'] == "TSP30"

    test_adata = adata[is_test].copy()

    # Copying the data so much uses an unreasonable amount of memory.
    trainval_adata = adata[~is_test].copy()

    return trainval_adata, test_adata

def loocv_donor_based(
    adata: sc.AnnData,
    label_key: str = "cell_type",
    group_key: str = "donor",
    random_state: int = 0
) -> List[Tuple[sc.AnnData, sc.AnnData]]:

    donors = adata.obs[group_key].unique()
    n_donors = len(donors)
    if n_donors < 2:
        raise ValueError(f"Need at least 2 unique groups; got {n_donors}")

    cv = StratifiedGroupKFold(n_splits=n_donors, shuffle=True, random_state=random_state)

    splits: List[Tuple[sc.AnnData, sc.AnnData]] = []
    for train_idx, val_idx in cv.split(
        X=adata.X,
        y=adata.obs[label_key],
        groups=adata.obs[group_key]
    ):
        train_adata = adata[train_idx].copy()
        val_adata   = adata[val_idx].copy()
        splits.append((train_adata, val_adata))

    return splits
    

def drop_annotations(adata: sc.AnnData) -> sc.AnnData:

    cols_to_drop = [
        'cell_ontology_id',
        'free_annotation',
        'manually_annotated'
    ]

    dropable = [c for c in cols_to_drop if c in adata.obs.columns]
    adata.obs.drop(columns=dropable, inplace=True)
    return adata

def save_processed(adata: sc.AnnData, out_filename: str)-> None:

    out_path = PROCESSED_DATA_DIR / out_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
