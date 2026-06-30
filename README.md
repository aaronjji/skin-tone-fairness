# SkinToneNet — Skin Tone-Aware Melanoma Detection

## Overview

SkinToneNet is an EfficientNet-B2 classifier trained on HAM10000 that conditions on a 3-dimensional ITA (Individual Typology Angle) skin tone encoding at inference time. The goal is to reduce the performance gap between light- and dark-skinned patients in automated melanoma detection.

**Key results (HAM10000 test set, 8-crop TTA):**
- Light skin (ITA > 41): AUC = 0.936
- Dark skin (ITA < 10): AUC = 0.869
- Gap: 6.7 percentage points (down from ~20pp reported in prior work on unmodified models)

## Repository Structure

```
skintone.py                  # Main training script (all variants)
run_inference.py             # HAM10000 test-set inference
run_ddi_inference.py         # DDI zero-shot inference
evaluate_ddi.py              # Full DDI evaluation with bootstrap CIs
generate_figures.py          # Generates all paper figures
download_datasets.py         # Helper scripts to download public datasets
skincancer_kaggle.ipynb      # Kaggle notebook (T4 GPU training)
results/                     # Predictions, ITA cache, split indices
ddi_results/                 # DDI evaluation outputs
figures/                     # Paper figures (PDF)
```

## Pretrained Weights

The four trained checkpoints are included in this repository under `results/`:

```
results/baseline_best.pt
results/aug_only_best.pt
results/tone_only_best.pt
results/full_best.pt
```

## Installation

```bash
pip install torch torchvision timm scikit-learn scipy pandas numpy Pillow tqdm matplotlib
```

Python 3.9+ recommended. Tested with PyTorch 2.x and numpy 2.x.

## Dataset Setup

Download HAM10000 via Kaggle:
```bash
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 --unzip -p data/ham10000
```

For DDI (Diverse Dermatology Images), request access at:
https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images

Expected layout:
```
data/
  ham10000/
    HAM10000_images_part_1/
    HAM10000_images_part_2/
    HAM10000_metadata.csv
  ddidiversedermatologyimages/
    ddi_metadata.csv
    <image files>
```

## Training

**On Kaggle (recommended — free T4 GPU):**
Open `skincancer_kaggle.ipynb` in Kaggle, attach the HAM10000 dataset and your `skintone.py` dataset, then run all cells. Training takes ~6 hours for all variants.

**Locally:**
```bash
# Train all four variants
python skintone.py --mode full --ham_dir data/ham10000 --output_dir results

# Train only specific variants
python skintone.py --mode full --variants baseline aug_only --ham_dir data/ham10000

# Train tone-balanced baseline (data-counting ablation)
python skintone.py --mode balanced --ham_dir data/ham10000 --output_dir results
```

Variants:
- `baseline` — EfficientNet-B2, standard augmentation
- `aug_only` — adds heavy augmentation (color jitter, elastic)
- `tone_only` — adds ITA conditioning, standard augmentation
- `full` — ITA conditioning + heavy augmentation (best model)

## Inference

```bash
# HAM10000 test set → results/test_predictions.csv
python run_inference.py

# DDI dataset → ddi_results/ddi_predictions.csv
python run_ddi_inference.py
```

## Evaluation

```bash
# Full DDI evaluation with bootstrap 95% CIs
python evaluate_ddi.py \
    --ddi_dir data/ddidiversedermatologyimages \
    --checkpoint_dir results \
    --output_dir ddi_results
```

## Figures

```bash
python generate_figures.py --out_dir figures
```

Generates four PDF figures for the paper.

## Citation

please contact my email if you need to cite

## License

Code: MIT License. Model weights trained on HAM10000 (CC BY-NC-SA 4.0) — non-commercial use only.
