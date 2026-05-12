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
BASE_OUT_DIR = Path("./synthetic_dataset/100_78_new_run")

GEN_PERCENT = 1.0  
GEN_SEED = 1908 

SPLIT_SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20
N_FOLDS = 5 

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
    if cls in PROMPTS_CACHE:
        return PROMPTS_CACHE[cls]
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

print("Loading original metadata for splits...")
df = pd.read_csv(ORIGINAL_METADATA_PATH)
df = df[df["diagnosis_3"].isin(DIAGNOSIS_MAP.keys())].copy()
df["diagnosis"] = df["diagnosis_3"].map(DIAGNOSIS_MAP)
df["label"] = df["diagnosis"].astype("category").cat.codes

lesion_df = (
    df.groupby("lesion_id")
    .agg(label=("label", lambda x: x.mode()[0]))
    .reset_index()
)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
folds = list(skf.split(lesion_df["lesion_id"], lesion_df["label"]))

print("\nLoading Stable Diffusion Model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

pipe.load_lora_weights(LORA_WEIGHTS)
negative_prompt = "clinical photo, patient skin, ruler, green, text, hair, watermark, low quality, blurry"

for current_fold in range(1, N_FOLDS + 1):
    print(f"\n{'='*40}")
    print(f" STARTING GENERATION FOR FOLD {current_fold}/{N_FOLDS}")
    print(f"{'='*40}")

    fold_dir = BASE_OUT_DIR / f"fold_{current_fold}"
    os.makedirs(fold_dir, exist_ok=True)

    trainval_idx, test_idx = folds[current_fold - 1]
    trainval_lesions = lesion_df.iloc[trainval_idx]

    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE) 
    train_lesions, val_lesions = train_test_split(
        trainval_lesions,
        test_size=val_fraction,
        random_state=SPLIT_SEED,
        stratify=trainval_lesions["label"],
    )

    train_df = df[df.lesion_id.isin(train_lesions["lesion_id"])]
    stats = train_df["diagnosis"].value_counts()
    print(f"Train targets for Fold {current_fold}:")
    print(stats)

    synthetic_metadata = []
    prompt_random = random.Random(GEN_SEED + current_fold)

    for cls in stats.index:
        target_count = int(stats[cls] * GEN_PERCENT)
        if target_count == 0:
            continue
            
        print(f"\n  -> Generating {target_count} images for [{cls}] in Fold {current_fold}")
        prompts_pool = load_prompts_for_class(cls)
        
        for i in tqdm(range(target_count), desc=cls):
            if prompts_pool:
                prompt = prompt_random.choice(prompts_pool)
            else:
                full_name = REV_DIAGNOSIS_MAP.get(cls, cls)
                prompt = f"dx_{cls.lower()}, {full_name.lower()}, dermoscopy image"

            with torch.no_grad(), torch.amp.autocast('cuda'):
                image = pipe(
                    prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=30,
                    guidance_scale=8.0,
                    cross_attention_kwargs={"scale": 0.7}
                ).images[0]

            synth_id = f"SYNTH_F{current_fold}_{cls}_{i:05d}"
            img_filename = f"{synth_id}.jpg"
            image.save(fold_dir / img_filename)

            new_row = {
                "isic_id": synth_id,
                "attribution": f"Synthetic (Stable Diffusion + LoRA) - Fold {current_fold}",
                "copyright_license": "CC-BY-NC",
                "age_approx": None,
                "anatom_site_1": "synthetic",
                "anatom_site_2": None,
                "anatom_site_general": None,
                "anatom_site_special": None,
                "concomitant_biopsy": False,
                "diagnosis_1": cls,
                "diagnosis_2": cls,
                "diagnosis_3": cls,
                "diagnosis_confirm_type": "synthetic_generation",
                "image_type": "dermoscopic",
                "lesion_id": f"SYNTH_LESION_F{current_fold}_{cls}_{i:05d}",
                "melanocytic": True if cls in ["MEL", "NV"] else False,
                "sex": None
            }
            synthetic_metadata.append(new_row)

    if synthetic_metadata:
        df_synth = pd.DataFrame(synthetic_metadata)
        csv_path = fold_dir / "metadata_synth.csv"
        df_synth.to_csv(csv_path, index=False)
        print(f"\nFold {current_fold} complete! Generated {len(df_synth)} images.")
        print(f"Saved to: {csv_path}")

    torch.cuda.empty_cache()
    gc.collect()

print("\n ALL FOLDS GENERATED SUCCESSFULLY!")