import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from prdc import compute_prdc
import gc
import open_clip


REAL_IMG_DIR = Path("./BCN_clean")
REAL_METADATA_PATH = Path("./metadata_clean.csv")
SYNTH_BASE_DIR = Path("./synthetic_dataset")
SYNTH_RUNS = ["78_3_run"] 

SEED = 1908
LOG_DIR = Path("./metrics_eval")
os.makedirs(LOG_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DIAGNOSIS_MAP = {
    "Solar or actinic keratosis": "AK", "Basal cell carcinoma": "BCC",
    "Seborrheic keratosis": "BKL", "Solar lentigo": "BKL",
    "Dermatofibroma": "DF", "Melanoma metastasis": "MEL",
    "Melanoma, NOS": "MEL", "Nevus": "NV",
    "Squamous cell carcinoma, NOS": "SCC",
}

REAL_FEATURES_CACHE_INCEPTION = {}
REAL_FEATURES_CACHE_BIOMED = {}
REAL_FEATURES_CACHE_CMMD = {} 

print("Loading InceptionV3...")
inception_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
inception_model.fc = torch.nn.Identity()
inception_model.to(device).eval()

inception_preprocess = transforms.Compose([
    transforms.Resize(299), transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading BiomedCLIP...")
biomed_id = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
biomed_model, _, biomed_preprocess = open_clip.create_model_and_transforms(biomed_id)
biomed_model.to(device).eval()

print("Loading CLIP ViT-L/14 (for CMMD)...")
cmmd_model, _, cmmd_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
cmmd_model.to(device).eval()

def get_inception_features(image_paths):
    feats = []
    with torch.no_grad():
        for p in tqdm(image_paths, desc="Inception feats", leave=False):
            img = Image.open(p).convert("RGB")
            tensor = inception_preprocess(img).unsqueeze(0).to(device)
            f = inception_model(tensor).cpu().numpy()
            feats.append(f)
    return np.vstack(feats)

def get_clip_features(image_paths, model_obj, preprocess_obj):
    feats = []
    batch_size = 32
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_tensors = [preprocess_obj(Image.open(p).convert("RGB")) for p in batch_paths]
            x = torch.stack(batch_tensors).to(device)
            f = model_obj.encode_image(x)
            f /= f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
    return np.vstack(feats)

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

def pil_to_uint8_tensor(path):
    img = Image.open(path).convert("RGB")
    return torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)


df_real_all = pd.read_csv(REAL_METADATA_PATH)
df_real_all["class"] = df_real_all["diagnosis_3"].map(DIAGNOSIS_MAP)

for run in SYNTH_RUNS:
    run_path = SYNTH_BASE_DIR / run
    metadata_path = run_path / "metadata_synth.csv"
    if not metadata_path.exists(): continue
    
    print(f"\n>>> Evaluating run: {run}")
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

        # 1. FID (Inception)
        n_fid = min(len(real_paths_all), len(synth_paths))
        real_paths_fid = [REAL_IMG_DIR / f"{rid}.jpg" for rid in df_cls_real_all.sample(n=n_fid, random_state=SEED)["isic_id"]]
        
        fid_metric = FrechetInceptionDistance(feature=2048).to(device)
        for p in real_paths_fid: fid_metric.update(pil_to_uint8_tensor(p).unsqueeze(0).to(device), real=True)
        for p in synth_paths: fid_metric.update(pil_to_uint8_tensor(p).unsqueeze(0).to(device), real=False)
        fid_v = fid_metric.compute().item()
        
        # 2. PRDC (Inception)
        if cls not in REAL_FEATURES_CACHE_INCEPTION:
            REAL_FEATURES_CACHE_INCEPTION[cls] = get_inception_features(real_paths_all)
        synth_feats_inc = get_inception_features(synth_paths)
        prdc_inc = compute_prdc(real_features=REAL_FEATURES_CACHE_INCEPTION[cls], fake_features=synth_feats_inc, nearest_k=5)
        
        # 3. PRDC (BiomedCLIP)
        if cls not in REAL_FEATURES_CACHE_BIOMED:
            REAL_FEATURES_CACHE_BIOMED[cls] = get_clip_features(real_paths_all, biomed_model, biomed_preprocess)
        synth_feats_bio = get_clip_features(synth_paths, biomed_model, biomed_preprocess)
        prdc_bio = compute_prdc(real_features=REAL_FEATURES_CACHE_BIOMED[cls], fake_features=synth_feats_bio, nearest_k=5)
        
        # 4. CMMD (CLIP ViT-L/14)
        if cls not in REAL_FEATURES_CACHE_CMMD:
            REAL_FEATURES_CACHE_CMMD[cls] = get_clip_features(real_paths_all, cmmd_model, cmmd_preprocess)
        synth_feats_cmmd = get_clip_features(synth_paths, cmmd_model, cmmd_preprocess)
        cmmd_v = compute_mmd(REAL_FEATURES_CACHE_CMMD[cls], synth_feats_cmmd)
        
        res = {
            "class": cls, "fid": fid_v, "cmmd": cmmd_v,
            **{f"inc_{k}": v for k, v in prdc_inc.items()},
            **{f"bio_{k}": v for k, v in prdc_bio.items()}
        }
        run_results.append(res)
        print(f"    FID: {fid_v:.2f} | CMMD: {cmmd_v:.4f} | Inc_R: {prdc_inc['recall']:.3f} | Bio_R: {prdc_bio['recall']:.3f}")
        
        torch.cuda.empty_cache()
        gc.collect()

    if run_results:
        pd.DataFrame(run_results).to_csv(LOG_DIR / f"metrics_{run}.csv", index=False)
