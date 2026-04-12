"""
run_ddi_inference.py
Runs full_best.pt on the DDI dataset and saves per-image scores.
Usage: python run_ddi_inference.py
Outputs: ddi_results/ddi_predictions.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score

from skintone import SkinToneNet, encode_ita, IMG_SIZE

# paths
BASE     = Path(".")
DDI_DIR  = BASE / "data" / "ddidiversedermatologyimages"
DDI_META = DDI_DIR / "ddi_metadata.csv"
CKPT     = BASE / "results" / "full_best.pt"
OUT_DIR  = BASE / "ddi_results"
OUT_CSV  = OUT_DIR / "ddi_predictions.csv"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# load model
print("Loading model...")
model = SkinToneNet(use_tone=True, pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()
print("Model loaded.")

# load metadata
meta = pd.read_csv(DDI_META)
meta.columns = [c.lower().strip() for c in meta.columns]

# FST packed codes → approximate ITA midpoint (Seité et al.)
FST_TO_ITA   = {12: 60.0, 34: 25.0, 56: -15.0,
                1: 65.0, 2: 55.0, 3: 30.0, 4: 20.0, 5: -10.0, 6: -20.0}
FST_TO_GROUP = {12: "light", 34: "medium", 56: "dark",
                1: "light", 2: "light", 3: "medium", 4: "medium",
                5: "dark",  6: "dark"}

# 8-crop TTA (same as skintone.py eval_tta)
tta_tfms = []
for rot in [0, 90, 180, 270]:
    for flip in [False, True]:
        ops = [T.Resize(256), T.CenterCrop(IMG_SIZE)]
        if rot:
            ops.append(T.RandomRotation((rot, rot)))
        if flip:
            ops.append(T.RandomHorizontalFlip(p=1.0))
        ops += [T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        tta_tfms.append(T.Compose(ops))

# inference
results = []
with torch.no_grad():
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="DDI inference"):
        img_path = DDI_DIR / row["ddi_file"]
        if not img_path.exists():
            continue

        fst        = row["skin_tone"]
        label      = int(row["malignant"])
        tone_group = FST_TO_GROUP.get(fst, "unknown")
        ita_val    = FST_TO_ITA.get(fst, 0.0)
        ita_enc    = torch.tensor(encode_ita(ita_val), dtype=torch.float32).unsqueeze(0).to(DEVICE)

        img = Image.open(img_path).convert("RGB")
        scores = [torch.sigmoid(model(tfm(img).unsqueeze(0).to(DEVICE), ita_enc)).item()
                  for tfm in tta_tfms]

        results.append({
            "ddi_file":   row["ddi_file"],
            "score":      float(np.mean(scores)),
            "label":      label,
            "skin_tone":  fst,
            "tone_group": tone_group,
            "ita":        ita_val,
        })

df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(df)} predictions → {OUT_CSV}")

# AUC + threshold sweep
print(f"\nOverall AUC: {roc_auc_score(df['label'], df['score']):.4f}")

print("\n=== THRESHOLD SWEEP (full DDI) ===")
for t in [0.30, 0.35, 0.40, 0.45, 0.50]:
    preds = (df["score"] >= t).astype(int)
    tp = ((preds==1) & (df["label"]==1)).sum()
    fn = ((preds==0) & (df["label"]==1)).sum()
    fp = ((preds==1) & (df["label"]==0)).sum()
    tn = ((preds==0) & (df["label"]==0)).sum()
    sens = tp/(tp+fn) if (tp+fn) else 0
    spec = tn/(tn+fp) if (tn+fp) else 0
    print(f"  t={t:.2f}: Sens={sens:.3f} Spec={spec:.3f}")

print("\n=== PER TONE GROUP (t=0.5) ===")
for group in ["light", "medium", "dark"]:
    sub = df[df["tone_group"] == group]
    if len(sub) < 5 or sub["label"].nunique() < 2:
        continue
    auc   = roc_auc_score(sub["label"], sub["score"])
    preds = (sub["score"] >= 0.5).astype(int)
    tp = ((preds==1) & (sub["label"]==1)).sum()
    fn = ((preds==0) & (sub["label"]==1)).sum()
    fp = ((preds==1) & (sub["label"]==0)).sum()
    tn = ((preds==0) & (sub["label"]==0)).sum()
    sens = tp/(tp+fn) if (tp+fn) else 0
    spec = tn/(tn+fp) if (tn+fp) else 0
    print(f"  {group}: n={len(sub)} AUC={auc:.4f} Sens={sens:.3f} Spec={spec:.3f}")
