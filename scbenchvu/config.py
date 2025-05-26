from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

V1_CODES = [f"TSP{i}" for i in range(1, 16)]
V2_CODES = ["TSP17", "TSP19", "TSP20", "TSP21", "TSP25", "TSP26", "TSP27", "TSP28", "TSP30"]

SCGPT_HYPERPARAMS ={
    "seed": 42,                        
    "dataset_name": "ovary_v2",        
    "load_model": MODELS_DIR / "scGPT_human",
    "do_train": True,                  
    "mask_ratio": 0.0,                 
    "n_bins": 10,                      
    "epochs": 2,                      
    "batch_size": 16,                  
    "lr": 1e-4,                        
    "layer_size": 128,                 
    "nlayers": 4,                      
    "nhead": 4,                        
    "dropout": 0.2,                    
    "save_eval_interval": 2,           
    "amp": True,                       
    "freeze": False,                   
}