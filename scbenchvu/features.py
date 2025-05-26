from scgpt.preprocess import Preprocessor
from .config import SCGPT_HYPERPARAMS

def bin_data(adata):
    pre = Preprocessor(
        use_key="X",
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=True,
        result_log1p_key="X_log1p",
        binning=SCGPT_HYPERPARAMS['n_bins'],
        result_binned_key="X_binned"
    )
    pre(adata)
    return adata

