import os
import gc
import torch
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from glob import glob
from diffusers import StableDiffusionPipeline
from sklearn.model_selection import StratifiedKFold, train_test_split


LORA_WEIGHTS = "./LoRa/lora_output/78_new_run/last.safetensors"
DATASET_LORA_DIR = "./datasets/bcn_org"
ORIGINAL_METADATA_PATH = "./metadata_clean.csv"
BASE_OUT_DIR = Path("./synthetic_dataset/nv_balanced_run")

REFERENCE_CLASS = "NV" 

GEN_SEED = 1908 
SPLIT_SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20
N_FOLDS = 5 

# Маппинг 
DIAGNOSIS_MAP = {
    "Solar or actinic keratosis": "AK",
    "Basal cell carcinoma": "BCC",
    "Seborrheic keratosis": "BKL",
    "Solar lentigo": "BKL",
    "Dermatofibroma": "DF",
    "Melanoma metastasis": "MEL",
    "Melanoma, NOS": "MEL",
    "Nevus": "NV",
    "Squamous cell carcinoma, NOS": "SCC",
}

REV_DIAGNOSIS_MAP = {v: k for k, v in DIAGNOSIS_MAP.items()}
PROMPTS_CACHE = {}

def load_prompts_for_class(cls):
    if cls in PROMPTS_CACHE: return PROMPTS_CACHE[cls]
    class_dir = os.path.join(DATASET_LORA_DIR, cls)
    if not os.path.exists(class_dir): return []
    txt_files = sorted(glob(os.path.join(class_dir, "*.txt")))
    prompts = []
    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                p = f.read().strip()
                if p: prompts.append(p)
        except Exception as e:
            print(f"Error {txt_file}: {e}")
    PROMPTS_CACHE[cls] = prompts
    return prompts

df = pd.read_csv(ORIGINAL_METADATA_PATH)
df = df[df["diagnosis_3"].isin(DIAGNOSIS_MAP.keys())].copy()
df["diagnosis"] = df["diagnosis_3"].map(DIAGNOSIS_MAP)
df["label"] = df["diagnosis"].astype("category").cat.codes

lesion_df = df.groupby("lesion_id").agg(label=("label", lambda x: x.mode()[0])).reset_index()
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
folds = list(skf.split(lesion_df["lesion_id"], lesion_df["label"]))

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
).to("cuda")
pipe.load_lora_weights(LORA_WEIGHTS)
negative_prompt = "clinical photo, patient skin, ruler, green, text, hair, watermark, low quality, blurry"

for fold_idx in range(N_FOLDS):
    current_fold = fold_idx + 1
    fold_dir = BASE_OUT_DIR / f"fold_{current_fold}"
    os.makedirs(fold_dir, exist_ok=True)

    trainval_idx, _ = folds[fold_idx]
    trainval_lesions = lesion_df.iloc[trainval_idx]
    train_lesions, _ = train_test_split(
        trainval_lesions, test_size=VAL_SIZE/(1.0-TEST_SIZE), 
        random_state=SPLIT_SEED, stratify=trainval_lesions["label"]
    )

    train_df = df[df.lesion_id.isin(train_lesions["lesion_id"])]
    stats = train_df["diagnosis"].value_counts()
    
    if REFERENCE_CLASS not in stats:
        continue
    
    target_count = stats[REFERENCE_CLASS]
    print(f"\n FOLD {current_fold} | Reference ({REFERENCE_CLASS}): {target_count}")

    synthetic_metadata = []
    prompt_random = random.Random(GEN_SEED + current_fold)

    for cls in stats.index:
        needed = target_count - stats[cls]
        
        if needed <= 0:
            continue
            
        print(f"  -> {cls}: generating {needed} (to {target_count})")
        prompts_pool = load_prompts_for_class(cls)
        
        for i in tqdm(range(needed), desc=f"F{current_fold} {cls}"):
            prompt = prompt_random.choice(prompts_pool) if prompts_pool else f"dx_{cls.lower()}, dermoscopy image"
            
            with torch.no_grad(), torch.amp.autocast('cuda'):
                image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30).images[0]

            synth_id = f"SYNTH_F{current_fold}_{cls}_{i:05d}"
            image.save(fold_dir / f"{synth_id}.jpg")

            synthetic_metadata.append({
                "isic_id": synth_id,
                "diagnosis_3": cls,
                "lesion_id": f"SYNTH_LESION_F{current_fold}_{cls}_{i:05d}",
                "fold": current_fold
            })

    if synthetic_metadata:
        pd.DataFrame(synthetic_metadata).to_csv(fold_dir / "metadata_synth.csv", index=False)

    torch.cuda.empty_cache()
    gc.collect()

print("\n Ready, all classes have " + REFERENCE_CLASS)