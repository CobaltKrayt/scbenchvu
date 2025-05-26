import scanpy as sc
import scgpt.tasks as tasks
from scbenchvu.dataset import drop_annotations
from sklearn.metrics import f1_score, classification_report
from scbenchvu.config import MODELS_DIR, PROCESSED_DATA_DIR

def run_inference():
    
    adata = sc.read_h5ad(PROCESSED_DATA_DIR / "ovary_test_1.h5ad")
    true_labels = adata.obs['cell_ontology_id'].values 
    adata_input = drop_annotations(adata.copy())

    emb_adata = tasks.embed_data(
        adata_input,
        MODELS_DIR / "scGPT_human",
        gene_col="index",
        obs_to_save=None,
        batch_size=16,
        return_new_adata=True,
        model_path="ovary_finetuned_ep10.pt",
        use_fast_transformer=True,       
        fast_transformer_backend="flash"
    )

    preds = tasks.annotate_data(
        emb_adata,
        model_path="ovary_finetuned_ep10.pt" 
    )

    f1_macro = f1_score(true_labels, preds, average='macro')  
    print(f"Macro F1 score: {f1_macro:.4f}")                         

    report = classification_report(true_labels, preds, 
                                   target_names=emb_adata.obs.columns)  
    print("Classification report:\n", report)                        

    adata.obs['predicted_cell_type'] = preds
    adata.write_h5ad(PROCESSED_DATA_DIR / "ovary_with_predictions.h5ad")
