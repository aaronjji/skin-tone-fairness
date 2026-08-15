# SkinToneNet — Skin Tone-Aware Melanoma Detection

**Code for: "Contrast-Induced Class Overlap as a Fairness Bottleneck in Dermatological AI: Evidence from HAM10000"**

Preprint: Research Square — https://doi.org/10.21203/rs.3.rs-10132969/v1

## Overview

SkinToneNet is an EfficientNet-B2 classifier trained on HAM10000 that conditions on a 3-dimensional ITA (Individual Typology Angle) skin tone encoding at inference time. The goal is to reduce the performance gap between light- and dark-skinned patients in automated melanoma detection. The paper refers to this architecture as **ITA-CondNet**.

**Key results (HAM10000 test set, high-confidence ITA subset, 8-crop TTA, full model):**
- Specificity deficit on darker skin: −11.0 pp (0.715 dark vs. 0.826 light), while sensitivity is comparable across tone (permutation p < 0.0002)
- Signal-to-noise ratio reduction from lighter to darker skin: 5.2×
- Per-tone class balancing (not tone conditioning) recovers the most dark-skin specificity: 0.668 → 0.799
- Translates to an estimated ≈70 excess unnecessary referrals per 1,000 darker-skin patients

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

All five trained checkpoints are included in this repository under `results/`:

```
results/baseline_best.pt
results/aug_only_best.pt
results/tone_only_best.pt
results/full_best.pt
results/baseline_balanced_best.pt   # Balanced Baseline — recovers the most dark-skin specificity
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
# Train the four ITA/augmentation variants
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
- `full` — ITA conditioning + heavy augmentation
- `baseline_balanced` (`--mode balanced`) — equalised class exposure (pos_weight=1.0), the data-counting control; recovers the most dark-skin specificity of any variant

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

Accepted at the FAIMI-BRIDGE-EPIMI 2026 MICCAI workshop; the Springer LNCS DOI will be
added here once assigned. In the meantime, cite the Research Square preprint above or the
workshop version below.

```bibtex
@inproceedings{ajit2026contrast,
  author    = {Ajit, Aaron},
  title     = {Contrast-Induced Class Overlap as a Fairness Bottleneck in
               Dermatological AI: Evidence from {HAM10000}},
  booktitle = {Fairness of AI in Medical Imaging (FAIMI) / Regulatory
               Evaluation (BRIDGE) / Ethics \& Philosophy (EPIMI) --
               Joint MICCAI 2026 Workshop},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026},
  note      = {To appear}
}
```

## License

Code: MIT License. Model weights trained on HAM10000 (CC BY-NC-SA 4.0) — non-commercial use only.
