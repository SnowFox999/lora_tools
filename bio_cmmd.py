import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import gc
import open_clip

REAL_IMG_DIR = Path("./BCN_clean")
REAL_METADATA_PATH = Path("./metadata_clean.csv")
SYNTH_BASE_DIR = Path("./synthetic_dataset")
SYNTH_RUNS = ["78_3_run"] 

SEED = 1908
LOG_DIR = Path("./bio_cmmd_eval")
os.makedirs(LOG_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DIAGNOSIS_MAP = {
    "Solar or actinic keratosis": "AK", "Basal cell carcinoma": "BCC",
    "Seborrheic keratosis": "BKL", "Solar lentigo": "BKL",
    "Dermatofibroma": "DF", "Melanoma metastasis": "MEL",
    "Melanoma, NOS": "MEL", "Nevus": "NV",
    "Squamous cell carcinoma, NOS": "SCC",
}

REAL_FEATURES_CACHE_BIOMED = {}

print("Loading BiomedCLIP for CMMD calculation...")
biomed_id = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
biomed_model, _, biomed_preprocess = open_clip.create_model_and_transforms(biomed_id)
biomed_model.to(device).eval()

def get_clip_features(image_paths, model_obj, preprocess_obj):
    feats = []
    batch_size = 32
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_tensors = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_tensors.append(preprocess_obj(img))
                except Exception as e:
                    print(f"Error loading {p}: {e}")
            
            if not batch_tensors: continue
            
            x = torch.stack(batch_tensors).to(device)
            f = model_obj.encode_image(x)
            f /= f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
    return np.vstack(feats) if feats else np.array([])

def compute_mmd(x, y, sigma=10.0):
    x = np.asarray(x)
    y = np.asarray(y)
    
    def gaussian_kernel(a, b, sigma):
        sq_dist = np.sum(a**2, axis=1).reshape(-1, 1) + np.sum(b**2, axis=1) - 2 * np.dot(a, b.T)
        return np.exp(-sq_dist / (2 * sigma**2))

    xx = gaussian_kernel(x, x, sigma)
    yy = gaussian_kernel(y, y, sigma)
    xy = gaussian_kernel(x, y, sigma)
    return np.mean(xx) + np.mean(yy) - 2 * np.mean(xy)


df_real_all = pd.read_csv(REAL_METADATA_PATH)
df_real_all["class"] = df_real_all["diagnosis_3"].map(DIAGNOSIS_MAP)

for run in SYNTH_RUNS:
    run_path = SYNTH_BASE_DIR / run
    metadata_path = run_path / "metadata_synth.csv"
    if not metadata_path.exists(): continue
    
    print(f"\n>>> Evaluating run (Biomed-CMMD only): {run}")
    df_synth = pd.read_csv(metadata_path)
    counts = df_synth["diagnosis_3"].value_counts()
    valid_classes = counts[counts > 100].index.tolist()
    
    run_results = []
    
    for cls in valid_classes:
        print(f"  Class: {cls} (Synth: {counts[cls]})")
        s_ids = df_synth[df_synth["diagnosis_3"] == cls]["isic_id"].tolist()
        synth_paths = [run_path / f"{sid}.jpg" for sid in s_ids]
        
        df_cls_real_all = df_real_all[df_real_all["class"] == cls].sort_values("isic_id")
        real_paths_all = [REAL_IMG_DIR / f"{rid}.jpg" for rid in df_cls_real_all["isic_id"]]
        real_paths_all = [p for p in real_paths_all if p.exists()]

        if cls not in REAL_FEATURES_CACHE_BIOMED:
            print(f"    Caching BiomedCLIP features for real {cls}...")
            REAL_FEATURES_CACHE_BIOMED[cls] = get_clip_features(real_paths_all, biomed_model, biomed_preprocess)
        
        synth_feats = get_clip_features(synth_paths, biomed_model, biomed_preprocess)
        
        if len(synth_feats) > 0 and len(REAL_FEATURES_CACHE_BIOMED[cls]) > 0:
            cmmd_v = compute_mmd(REAL_FEATURES_CACHE_BIOMED[cls], synth_feats)
        else:
            cmmd_v = np.nan
        
        res = {
            "class": cls,
            "bio_cmmd": cmmd_v
        }
        run_results.append(res)
        print(f"    Biomed-CMMD: {cmmd_v:.5f}")
        
        torch.cuda.empty_cache()
        gc.collect()

    if run_results:
        out_path = LOG_DIR / f"biocmmd_{run}.csv"
        pd.DataFrame(run_results).to_csv(out_path, index=False)
        print(f"Results saved to {out_path}")
