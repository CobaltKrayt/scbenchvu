# src/<module_name>/modeling/train.py
import torch
import copy
from torch.optim import AdamW
from scgpt.trainer import prepare_data, prepare_dataloader, train, evaluate
from scbenchvu.dataset import load_and_filter, holdout_TSP30, save_processed
from scbenchvu.features import bin_data
from scbenchvu.config import MODELS_DIR, SCGPT_HYPERPARAMS
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from sklearn.model_selection import train_test_split

def fine_tune():
    # Load, filter, and preprocess
    adata = load_and_filter("Ovary_TSP1_30_version2d_10X_smartseq_scvi_Nov262024.h5ad")
    trainval_adata, test_adata = holdout_TSP30(adata)
    save_processed(test_adata, "ovary_test_1.h5ad")
    trainval_adata = bin_data(trainval_adata)
 

    # Split data and prepare loaders
    train_data, val_data = prepare_data(trainval_adata, split_ratio=0.8, label_key="cell_ontology_id")
    train_loader = prepare_dataloader(train_data, SCGPT_HYPERPARAMS['batch_size'], shuffle=True)
    val_loader   = prepare_dataloader(val_data,   SCGPT_HYPERPARAMS['batch_size'], shuffle=False)

    # Load model architecture and weights
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab  = GeneVocab.from_file(f"{MODELS_DIR}/scGPT_human/vocab.json")
    model  = TransformerModel(
        ntokens=len(vocab),
        embsize=SCGPT_HYPERPARAMS['layer_size'],
        nhead=SCGPT_HYPERPARAMS['nhead'],
        d_hid=SCGPT_HYPERPARAMS['layer_size'],
        nlayers=SCGPT_HYPERPARAMS['nlayers'],
        nlayers_cls=3,
        n_cls=adata.obs['cell_ontology_id'].nunique(),
        vocab=vocab,
        dropout=SCGPT_HYPERPARAMS['dropout'],
        pad_token=vocab.pad_token,
        pad_value=SCGPT_HYPERPARAMS['n_bins'],
        cell_emb_style="cls",    
        use_fast_transformer=True,
        fast_transformer_backend="flash",
    ).to(device)
    state = torch.load(f"{MODELS_DIR}/scGPT_human/best_model.pt", map_location=device)
    model.load_state_dict(state, strict=False)
    model.train()

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=SCGPT_HYPERPARAMS['lr'])

    # Training loop
    best_model, best_loss = copy.deepcopy(model.state_dict()), float('inf')
    for epoch in range(1, SCGPT_HYPERPARAMS['epochs'] + 1):
        train(model, train_loader, optimizer, device=device)
        val_loss, _ = evaluate(model, val_loader, device=device)
        print(f"Epoch {epoch} → Val Loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss, best_model = val_loss, copy.deepcopy(model.state_dict())
        if epoch % 5 == 0:
            torch.save(best_model, f"models/ovary_finetuned_ep{epoch}.pt")

if __name__ == "__main__":
    fine_tune()
