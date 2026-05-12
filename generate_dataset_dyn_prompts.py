import os
import torch
import pandas as pd
import random
from tqdm import tqdm
from pathlib import Path
from diffusers import StableDiffusionPipeline
from PIL import Image
from glob import glob

LORA_WEIGHTS = "./lora_output/78_new_run/last.safetensors"
DATASET_LORA_DIR = "./datasets/bcn_org"
ORIGINAL_METADATA_PATH = "./metadata_clean.csv"
OUT_DIR = Path("./synthetic_dataset")
OUT_IMG_DIR = OUT_DIR / "negative_run"

os.makedirs(OUT_IMG_DIR, exist_ok=True)

GEN_PERCENT = 0.2
SEED = 1908

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
    if not os.path.exists(class_dir):
        return []
    
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

print("Loading Model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

pipe.load_lora_weights(LORA_WEIGHTS)

df_orig = pd.read_csv(ORIGINAL_METADATA_PATH)
df_orig["class"] = df_orig["diagnosis_3"].map(DIAGNOSIS_MAP)
df_orig = df_orig.dropna(subset=["class"])

stats = df_orig["class"].value_counts()
all_class_ids = stats.index.tolist() 

print("\nOriginal dataset stats:")
print(stats)

synthetic_metadata = []

BASE_NEGATIVE_PROMPT = "clinical photo, patient skin, ruler, green, text, hair, watermark, low quality, blurry"

prompt_random = random.Random(SEED)

for cls in all_class_ids:
    target_count = int(stats[cls] * GEN_PERCENT)
    print(f"\n Generating {target_count} images for class: {cls}")
 
    other_class_tokens = [f"dx_{c.lower()}" for c in all_class_ids if c != cls]

    current_negative_prompt = f"{BASE_NEGATIVE_PROMPT}, {', '.join(other_class_tokens)}"
    
    prompts_pool = load_prompts_for_class(cls)
    
    for i in tqdm(range(target_count)):
        if prompts_pool:
            prompt = prompt_random.choice(prompts_pool)
        else:
            full_name = REV_DIAGNOSIS_MAP.get(cls, cls)
            prompt = f"dx_{cls.lower()}, {full_name.lower()}, dermoscopy image"

        with torch.no_grad(), torch.amp.autocast('cuda'):
            image = pipe(
                prompt,
                negative_prompt=current_negative_prompt, 
                num_inference_steps=30,
                guidance_scale=7.0,
                cross_attention_kwargs={"scale": 0.7}
            ).images[0]

        synth_id = f"SYNTH_{cls}_{i:05d}"
        img_filename = f"{synth_id}.jpg"
        image.save(OUT_IMG_DIR / img_filename)

        new_row = {
            "isic_id": synth_id,
            "attribution": "Synthetic (Stable Diffusion + LoRA)",
            "copyright_license": "CC-BY-NC",
            "age_approx": None,
            "anatom_site_1": "synthetic",
            "diagnosis_1": cls,
            "diagnosis_2": cls,
            "diagnosis_3": cls,
            "diagnosis_confirm_type": "synthetic_generation",
            "image_type": "dermoscopic",
            "lesion_id": f"SYNTH_LESION_{cls}_{i:05d}",
            "melanocytic": True if cls in ["MEL", "NV"] else False,
            "sex": None
        }
        synthetic_metadata.append(new_row)

    torch.cuda.empty_cache()

df_synth = pd.DataFrame(synthetic_metadata)
df_synth.to_csv(OUT_IMG_DIR / "metadata_synth.csv", index=False)

print(f"\n Done! Generated {len(df_synth)} images.")