"""
run_inference.py
Runs full_best.pt on the HAM10000 test set and saves per-image scores.
Usage: python run_inference.py
Outputs: results/test_predictions.csv
"""

import json, os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm

from skintone import SkinToneNet, encode_ita, IMG_SIZE

# paths
BASE      = Path(".")
DATA      = BASE / "data" / "ham10000"
IMG_DIRS  = [DATA / "HAM10000_images_part_1", DATA / "HAM10000_images_part_2"]
META_CSV  = DATA / "HAM10000_metadata.csv"
CKPT      = BASE / "results" / "full_best.pt"
ITA_CACHE = BASE / "results" / "ita_cache.json"
SPLIT_IDX = BASE / "results" / "split_indices.json"
OUT_CSV   = BASE / "results" / "test_predictions.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# load model
print("Loading model...")
model = SkinToneNet(use_tone=True, pretrained=False).to(DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(ckpt)
model.eval()
print("Model loaded OK.")

# load data
meta = pd.read_csv(META_CSV)
meta.columns = [c.lower().strip() for c in meta.columns]

with open(ITA_CACHE) as f:
    ita_cache = json.load(f)
with open(SPLIT_IDX) as f:
    splits = json.load(f)

# build image_id → path map
img_map = {}
for d in IMG_DIRS:
    for p in Path(d).iterdir():
        img_map[p.stem] = p

MALIGNANT = {"mel", "bcc", "akiec", "vasc"}
meta["malignant"] = meta["dx"].str.lower().str.strip().isin(MALIGNANT).astype(int)
id_to_label = dict(zip(meta["image_id"], meta["malignant"]))

test_ids = [i for i in splits["test"] if i in img_map]
print(f"Test images found: {len(test_ids)}")

# TTA transforms (8-crop, same as skintone.py eval_tta)
import torchvision.transforms as T

tta_tfms = []
for rot in [0, 90, 180, 270]:
    for flip in [False, True]:
        ops = [T.Resize(256), T.CenterCrop(IMG_SIZE)]
        if rot:
            ops.append(T.RandomRotation((rot, rot)))
        if flip:
            ops.append(T.RandomHorizontalFlip(p=1.0))
        ops += [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        tta_tfms.append(T.Compose(ops))

# inference
results = []
with torch.no_grad():
    for image_id in tqdm(test_ids, desc="Inference"):
        img = Image.open(img_map[image_id]).convert("RGB")
        ita_val = ita_cache.get(image_id) or 0.0
        ita_enc = torch.tensor(encode_ita(ita_val), dtype=torch.float32).unsqueeze(0).to(DEVICE)

        scores = []
        for tfm in tta_tfms:
            x = tfm(img).unsqueeze(0).to(DEVICE)
            scores.append(torch.sigmoid(model(x, ita_enc)).item())

        results.append({
            "image_id": image_id,
            "score":    float(np.mean(scores)),
            "label":    id_to_label.get(image_id, -1),
            "ita":      ita_val,
        })

df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(df)} predictions → {OUT_CSV}")

# quick AUC check
from sklearn.metrics import roc_auc_score
valid = df[df["label"] >= 0]
print(f"Overall AUC: {roc_auc_score(valid['label'], valid['score']):.4f}")
dark  = valid[valid["ita"] < 10]
light = valid[valid["ita"] > 41]
if len(light) > 0:
    print(f"Light (ITA>41): n={len(light)}, AUC={roc_auc_score(light['label'], light['score']):.4f}")
if len(dark) > 0:
    print(f"Dark  (ITA<10): n={len(dark)},  AUC={roc_auc_score(dark['label'],  dark['score']):.4f}")
