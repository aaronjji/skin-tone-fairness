"""
download_datasets.py
====================
Download public skin-cancer / dermatology datasets relevant to skin-tone
fairness research. Covers 8 databases with different access methods.

QUICK START
-----------
# See what's available
python download_datasets.py --list

# Download one dataset
python download_datasets.py --dataset isic2020 --out_dir ./data

# Download all no-registration datasets
python download_datasets.py --dataset all_free --out_dir ./data

DEPENDENCIES
------------
pip install requests tqdm kaggle pandas

DATASETS OVERVIEW
-----------------
Dataset         | Images  | Skin-Tone Labels | Registration
----------------|---------|------------------|--------------
isic2020        | 33,126  | No               | None (free)
isic2019        | 25,331  | No               | None (free)
isic2018        | 15,414  | No               | None (free)
pad_ufes20      |  2,298  | Fitzpatrick 1-6  | None (free)
bcn20000        | 18,946  | No               | None (free)
sd198           |  6,584  | No               | Kaggle account
fitzpatrick17k  | 16,577  | Fitzpatrick I-VI | Email request
ddi             |    656  | Fitzpatrick I-VI | Stanford agreement
derm7pt         |  2,000+ | No               | SFU password form
"""

import os
import sys
import json
import argparse
import hashlib
import zipfile
import tarfile
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

# optional — installed lazily below
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# Progress Bar Helper


class _Progress:
    """Simple download progress hook for urlretrieve."""
    def __init__(self, desc="Downloading"):
        self.desc = desc
        self._pbar = None

    def __call__(self, block_num, block_size, total_size):
        if HAS_TQDM:
            if self._pbar is None:
                self._pbar = tqdm(total=total_size, unit="B", unit_scale=True,
                                  desc=self.desc)
            downloaded = block_num * block_size
            self._pbar.update(min(block_size, total_size - self._pbar.n))
            if downloaded >= total_size and self._pbar:
                self._pbar.close()
        else:
            pct = min(100, int(block_num * block_size * 100 / max(total_size, 1)))
            print(f"\r{self.desc}: {pct}%", end="", flush=True)


def _download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Download url → dest. Returns dest path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] already exists: {dest.name}")
        return dest
    print(f"  Downloading {desc or dest.name} ...")
    try:
        urlretrieve(url, dest, reporthook=_Progress(desc or dest.name))
    except URLError as e:
        print(f"\n  [ERROR] Download failed: {e}")
        raise
    if not HAS_TQDM:
        print()
    return dest


def _extract(archive: Path, dest_dir: Path):
    """Extract zip or tar archive."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    print(f"  Extracting {archive.name} → {dest_dir} ...")
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as z:
            z.extractall(dest_dir)
    elif name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar")):
        with tarfile.open(archive, "r:*") as t:
            t.extractall(dest_dir)
    else:
        print(f"  [WARN] Unknown archive format: {archive.name} — skipping extract")


# ISIC Archive — no registration required
# REST API docs: https://api.isic-archive.com/api/docs/swagger/
# AWS S3: arn:aws:s3:::isic-archive
ISIC_CHALLENGE_URLS = {
    "isic2016": {
        "train_images": "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3B_Training_Data.zip",
        "test_images":  "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3B_Test_Data.zip",
        "train_labels": "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv",
        "test_labels":  "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3B_Test_GroundTruth.csv",
    },
    "isic2017": {
        "train_images": "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip",
        "val_images":   "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Data.zip",
        "test_images":  "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Data.zip",
        "train_labels": "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part3_GroundTruth.csv",
        "val_labels":   "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part3_GroundTruth.csv",
        "test_labels":  "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv",
    },
    "isic2018": {
        "train_images": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip",
        "val_images":   "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_Input.zip",
        "test_images":  "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_Input.zip",
        "train_labels": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip",
    },
    "isic2019": {
        "train_images": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip",
        "train_labels": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv",
        "train_meta":   "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv",
    },
    "isic2020": {
        "train_images": "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip",
        "test_images":  "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_JPEG.zip",
        "train_labels": "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth.csv",
        "train_meta":   "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_Metadata.csv",
        "test_meta":    "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_Metadata.csv",
    },
}


def download_isic_challenge(year: str, out_dir: Path):
    """Download an ISIC challenge dataset (2016-2020)."""
    key = f"isic{year}"
    if key not in ISIC_CHALLENGE_URLS:
        print(f"[ERROR] Unknown ISIC year: {year}. Choose 2016/2017/2018/2019/2020")
        return

    dest = out_dir / f"isic{year}"
    dest.mkdir(parents=True, exist_ok=True)
    urls = ISIC_CHALLENGE_URLS[key]

    print(f"\n{'='*60}")
    print(f" ISIC {year} Challenge Dataset")
    print(f"{'='*60}")

    for name, url in urls.items():
        fname = dest / Path(url).name
        _download_file(url, fname, name)
        if fname.suffix == ".zip":
            _extract(fname, dest / name)

    print(f"\n  Done. Files in: {dest}")
    _write_readme(dest, key, ISIC_CHALLENGE_URLS[key])


def download_isic_via_api(out_dir: Path, limit: int = 5000, offset: int = 0):
    """
    Download images + metadata from the ISIC REST API.
    No authentication required for public images.
    API docs: https://api.isic-archive.com/api/docs/swagger/
    """
    if not HAS_REQUESTS:
        print("[ERROR] Install requests: pip install requests")
        return

    dest = out_dir / "isic_api"
    dest.mkdir(parents=True, exist_ok=True)
    img_dir = dest / "images"
    img_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f" ISIC Archive API  (limit={limit}, offset={offset})")
    print(f"{'='*60}")
    print("  API: https://api.isic-archive.com")

    meta_path = dest / "metadata.jsonl"
    base = "https://api.isic-archive.com/api/v2/images/"
    params = {"limit": min(limit, 200), "offset": offset,
              "format": "json", "sort": "created"}

    downloaded = 0
    page = 0
    with open(meta_path, "a") as fmeta:
        while downloaded < limit:
            params["offset"] = offset + downloaded
            params["limit"] = min(200, limit - downloaded)
            try:
                r = requests.get(base, params=params, timeout=30)
                r.raise_for_status()
            except requests.RequestException as e:
                print(f"  [ERROR] API request failed: {e}")
                break

            data = r.json()
            results = data.get("results", [])
            if not results:
                break

            for item in results:
                isic_id = item.get("isic_id", "")
                fmeta.write(json.dumps(item) + "\n")

                # Download image
                img_path = img_dir / f"{isic_id}.jpg"
                if img_path.exists():
                    continue
                img_url = item.get("files", {}).get("full", {}).get("url")
                if img_url:
                    try:
                        ir = requests.get(img_url, timeout=30)
                        ir.raise_for_status()
                        img_path.write_bytes(ir.content)
                    except requests.RequestException:
                        pass  # skip failed images

            downloaded += len(results)
            page += 1
            print(f"  Page {page}: {downloaded}/{limit} images", end="\r")

            if not data.get("next"):
                break

    print(f"\n  Done. Metadata: {meta_path}")
    print(f"  Images: {img_dir}")


# PAD-UFES-20 — no registration, CC BY
# Mendeley Data: https://data.mendeley.com/datasets/zr7vgbcyr2/1
# Contains Fitzpatrick skin type labels (1-6)
PAD_UFES_FILES = {
    # Direct Mendeley download links (v1 dataset)
    "metadata":"https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/3d3d4c2f-42d0-416a-9a23-5b6ce21e5c08/file_downloaded",
    "images_1":  "https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/38ebe0ec-3b10-4afe-9a8e-8cb5e6e1c08c/file_downloaded",
    "images_2":  "https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/48a3e2b3-cb2a-445a-9fdb-b85a6898ed3a/file_downloaded",
    "images_3":  "https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/c1db7a78-db0e-4aaf-bef1-b6b66bebd2a6/file_downloaded",
}

PAD_LABELS = {
    "BCC": "Basal Cell Carcinoma (malignant)",
    "SCC": "Squamous Cell Carcinoma (malignant)",
    "MEL": "Melanoma (malignant)",
    "ACK": "Actinic Keratosis (pre-malignant)",
    "BOD": "Bowen's Disease (pre-malignant)",
    "SEK": "Seborrheic Keratosis (benign)",
    "NEV": "Nevus (benign)",
}


def download_pad_ufes20(out_dir: Path):
    """
    Download PAD-UFES-20 — 2,298 images with Fitzpatrick skin type labels.
    No registration required.  License: CC BY.
    Publication: https://doi.org/10.1016/j.dib.2020.106221
    """
    dest = out_dir / "pad_ufes20"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" PAD-UFES-20 (Brazilian, Fitzpatrick-labeled)")
    print(f"{'='*60}")
    print("  Diagnosis classes: BCC, SCC, MEL, ACK, BOD, SEK, NEV")
    print("  Skin tone: Fitzpatrick 1-6 labels included in metadata")
    print("  Mendeley: https://data.mendeley.com/datasets/zr7vgbcyr2/1\n")

    rename_map = {
        "metadata": "metadata.csv",
        "images_1":  "images_part1.zip",
        "images_2":  "images_part2.zip",
        "images_3":  "images_part3.zip",
    }

    for key, url in PAD_UFES_FILES.items():
        fname = dest / rename_map[key]
        try:
            _download_file(url, fname, key)
            if fname.suffix == ".zip":
                _extract(fname, dest / "images")
        except Exception as e:
            print(f"\n  [WARN] {key} download failed: {e}")
            print(f"  Manual download: https://data.mendeley.com/datasets/zr7vgbcyr2/1")

    # Write label reference
    (dest / "label_reference.json").write_text(json.dumps(PAD_LABELS, indent=2))
    print(f"\n  Done. Files in: {dest}")
    print("  Key column in metadata.csv: 'fitzpatrick' (values 1-6)")


# BCN20000 — no registration, CC-BY 4.0
# Hospital Clínic Barcelona — 18,946 dermoscopy images
# Paper: https://www.nature.com/articles/s41597-024-03387-w
BCN_FIGSHARE_BASE = "https://figshare.com/ndownloader/articles/24893295/versions/1"
def download_bcn20000(out_dir: Path):
    """
    Download BCN20000 from Figshare.  License: CC-BY 4.0.
    ~18,946 images, age + sex + diagnosis metadata.
    Note: no Fitzpatrick labels, but all images from Mediterranean population.
    """
    dest = out_dir / "bcn20000"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" BCN20000 (Hospital Clínic Barcelona)")
    print(f"{'='*60}")
    print("  Images: 18,946  |  License: CC-BY 4.0")
    print("  No Fitzpatrick labels; age, sex, diagnosis available")
    print("  Paper: https://www.nature.com/articles/s41597-024-03387-w\n")

    archive = dest / "bcn20000_figshare.zip"
    try:
        _download_file(BCN_FIGSHARE_BASE, archive, "BCN20000 (Figshare)")
        _extract(archive, dest)
    except Exception as e:
        print(f"\n  [WARN] Automatic download failed: {e}")
        print("  Manual download:")
        print("    https://figshare.com/articles/dataset/BCN20000/24893295")
        print("  GitHub (metadata only):")
        print("    git clone https://github.com/imatge-upc/BCN20000")

    print(f"\n  Done (check {dest} for contents).")


# SD-198 — via Kaggle, requires free Kaggle account
# 6,584 images across 198 skin disease categories
def download_sd198(out_dir: Path):
    """
    Download SD-198 via Kaggle CLI.
    Requires: kaggle.json in ~/.kaggle/
    Get credentials: https://www.kaggle.com/account → Settings → API → Create Token
    """
    import subprocess

    dest = out_dir / "sd198"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" SD-198  (198 skin disease categories, 6,584 images)")
    print(f"{'='*60}")
    print("  Requires: Kaggle account + ~/.kaggle/kaggle.json")
    print("  Get token: https://www.kaggle.com/account → Settings → API\n")

    try:
        subprocess.run(
            ["kaggle", "datasets", "download",
             "-d", "longngzzz/sd-198",
             "--unzip", "-p", str(dest)],
            check=True
        )
        print(f"\n  Done. Files in: {dest}")
    except subprocess.CalledProcessError:
        print("\n  [ERROR] Kaggle download failed.")
        print("  Manual: https://www.kaggle.com/datasets/longngzzz/sd-198")
    except FileNotFoundError:
        print("\n  [ERROR] 'kaggle' CLI not found. Install: pip install kaggle")


# HAM10000 — via Kaggle, same dataset as skintone.py pipeline
def download_ham10000(out_dir: Path):
    """Download HAM10000 via Kaggle CLI (same dataset as skintone.py uses)."""
    import subprocess

    dest = out_dir / "ham10000"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" HAM10000 (10,015 images, 7 diagnostic classes)")
    print(f"{'='*60}")
    print("  Requires: Kaggle account + ~/.kaggle/kaggle.json")
    print("  Get token: https://www.kaggle.com/account → Settings → API\n")

    try:
        subprocess.run(
            ["kaggle", "datasets", "download",
             "-d", "kmader/skin-cancer-mnist-ham10000",
             "--unzip", "-p", str(dest)],
            check=True
        )
        print(f"\n  Done. Files in: {dest}")
    except subprocess.CalledProcessError:
        print("\n  [ERROR] Kaggle download failed.")
        print("  Manual: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
    except FileNotFoundError:
        print("\n  [ERROR] 'kaggle' CLI not found. Install: pip install kaggle")


# Fitzpatrick17k — email request required, best dataset for skin-tone research
# 16,577 images with Fitzpatrick I-VI labels
FITZPATRICK17K_CSV_URL = (
    "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/"
    "main/fitzpatrick17k.csv"
)


def download_fitzpatrick17k(out_dir: Path):
    """
    Download the Fitzpatrick17k metadata CSV (always available).
    Full images require contacting the authors via GitHub form.
    Repo: https://github.com/mattgroh/fitzpatrick17k
    License: CC BY-NC 4.0
    """
    dest = out_dir / "fitzpatrick17k"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Fitzpatrick17k  (16,577 images, Fitzpatrick I-VI labels)")
    print(f"{'='*60}")
    print("  BEST dataset for skin-tone fairness research")
    print("  License: CC BY-NC 4.0")
    print("  Repo: https://github.com/mattgroh/fitzpatrick17k\n")

    # Download the metadata CSV (always public)
    csv_dest = dest / "fitzpatrick17k.csv"
    try:
        _download_file(FITZPATRICK17K_CSV_URL, csv_dest, "fitzpatrick17k metadata CSV")
        print(f"\n  CSV downloaded: {csv_dest}")
        print("  Columns include: url, fitzpatrick_scale, label, three_partition_label")
    except Exception as e:
        print(f"  [WARN] CSV download failed: {e}")

    # Try bulk image download from the CSV URLs
    print("\n  Attempting image download from CSV URLs ...")
    print("  (Many source URLs may be broken; contact authors for full dataset)")
    _try_fitzpatrick17k_images(csv_dest, dest / "images")

    # Instructions for full dataset
    print("\n" + "─"*60)
    print("  For the complete dataset, contact authors:")
    print("  1. Open: https://github.com/mattgroh/fitzpatrick17k")
    print("  2. Submit the Google Form in the README")
    print("  3. You'll receive a Google Drive / Dropbox link by email")
    print("─"*60)

    _write_readme(dest, "fitzpatrick17k", {
        "repo": "https://github.com/mattgroh/fitzpatrick17k",
        "license": "CC BY-NC 4.0",
        "images": 16577,
        "skin_tone": "Fitzpatrick I-VI",
        "note": "Full images require author contact via GitHub form",
    })


def _try_fitzpatrick17k_images(csv_path: Path, img_dir: Path, max_images: int = 200):
    """Try downloading up to max_images from the CSV URL column."""
    if not csv_path.exists():
        return
    try:
        import csv
        img_dir.mkdir(exist_ok=True)
        saved = 0
        failed = 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if saved >= max_images:
                    break
                url = row.get("url", "").strip()
                if not url:
                    continue
                # derive filename from URL
                fname = hashlib.md5(url.encode()).hexdigest()[:12] + ".jpg"
                dest = img_dir / fname
                if dest.exists():
                    saved += 1
                    continue
                if HAS_REQUESTS:
                    try:
                        r = requests.get(url, timeout=10)
                        if r.status_code == 200:
                            dest.write_bytes(r.content)
                            saved += 1
                        else:
                            failed += 1
                    except Exception:
                        failed += 1
        print(f"  Images saved: {saved}  |  failed/broken URLs: {failed}")
        if saved < 100:
            print("  Most source URLs appear broken — contact authors for full dataset")
    except Exception as e:
        print(f"  [WARN] Image download attempt failed: {e}")


# DDI — Stanford Research Use Agreement required
# 656 biopsy-confirmed images, Fitzpatrick I-VI, most diverse benchmark
def download_ddi(out_dir: Path):
    """
    Print registration instructions for DDI (Diverse Dermatology Images).
    Images require a Stanford Research Use Agreement.
    Dataset: https://ddi-dataset.github.io/
    Portal:  https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images
    """
    dest = out_dir / "ddi"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" DDI — Diverse Dermatology Images  (Stanford)")
    print(f"{'='*60}")
    print("  656 biopsy-confirmed images | Fitzpatrick I-VI | CC Non-Commercial")
    print("  All images pathology-confirmed — gold standard test set\n")
    print("  HOW TO GET ACCESS:")
    print("  1. Visit: https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images")
    print("  2. Click 'Request Access'")
    print("  3. Sign Stanford Research Use Agreement")
    print("  4. Download link provided via email (~hours)\n")
    print("  Also see DDI-2 (2024) — 665 images from Asian patients:")
    print("  https://aimi.stanford.edu (search DDI-2)")
    print("─"*60)

    _write_readme(dest, "ddi", {
        "portal": "https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images",
        "images": 656,
        "skin_tone": "Fitzpatrick I-VI (balanced)",
        "license": "Stanford Research Use Agreement (non-commercial)",
        "access": "Sign agreement at Stanford AIMI portal",
    })


# Derm7pt — SFU password form, instant email
# 2,000+ images, 7-point checklist labels
def download_derm7pt(out_dir: Path):
    """Print registration instructions for Derm7pt."""
    dest = out_dir / "derm7pt"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Derm7pt  (7-point checklist, SFU)")
    print(f"{'='*60}")
    print("  ~2,000 images | License: CC BY-NC-ND 4.0")
    print("  Labels: 7-point checklist criteria + diagnosis class\n")
    print("  HOW TO GET ACCESS:")
    print("  1. Visit: https://derm.cs.sfu.ca/Download.html")
    print("  2. Fill out the request form")
    print("  3. Password emailed immediately (automated)")
    print("  4. Download zip from: http://derm.cs.sfu.ca")
    print("─"*60)

    _write_readme(dest, "derm7pt", {
        "download_page": "https://derm.cs.sfu.ca/Download.html",
        "images": "~2000",
        "license": "CC BY-NC-ND 4.0",
        "access": "Instant password via email form at SFU",
    })


# Utility


def _write_readme(dest: Path, name: str, info: dict):
    readme = dest / "DATASET_INFO.json"
    readme.write_text(json.dumps({"dataset": name, **info}, indent=2))


def print_summary():
    """Print a comparison table of all datasets."""
    rows = [
        ("Dataset",        "Images", "Skin-Tone Labels", "Registration",    "License"),
        ("─"*14,           "─"*7,    "─"*18,            "─"*16,            "─"*14),
        ("HAM10000",       "10,015", "No (ITA computable)", "Kaggle acct",  "CC BY-NC-SA"),
        ("ISIC 2020",      "33,126", "No",               "None",            "CC-0/BY/BY-NC"),
        ("ISIC 2019",      "25,331", "No",               "None",            "CC-0/BY/BY-NC"),
        ("ISIC 2018",      "15,414", "No",               "None",            "CC-0/BY/BY-NC"),
        ("PAD-UFES-20",    " 2,298", "Fitzpatrick 1-6",  "None",            "CC BY"),
        ("BCN20000",       "18,946", "No",               "None",            "CC-BY 4.0"),
        ("SD-198",         " 6,584", "No",               "Kaggle acct",     "Open"),
        ("Fitzpatrick17k", "16,577", "Fitzpatrick I-VI", "Email request",   "CC BY-NC"),
        ("DDI",            "   656", "Fitzpatrick I-VI", "Stanford agr.",   "Non-commercial"),
        ("Derm7pt",        " 2,000", "No",               "Instant form",    "CC BY-NC-ND"),
    ]
    print("\n" + "="*80)
    print(" SKIN CANCER DATASET COMPARISON")
    print("="*80)
    for row in rows:
        print(f"  {row[0]:<16} {row[1]:<8} {row[2]:<20} {row[3]:<18} {row[4]}")
    print("="*80)
    print("\n  RECOMMENDED COMBINATION for skin-tone fairness research:")
    print("  ► Fitzpatrick17k  — training/pre-training (skin tone I-VI)")
    print("  ► PAD-UFES-20     — labeled skin tone, Brazilian population")
    print("  ► DDI             — held-out test set (biopsy-confirmed, balanced FST)")
    print("  ► ISIC 2020       — large unlabeled pool (compute ITA yourself)")
    print("  ► HAM10000        — your existing pipeline baseline\n")


# Cli


ALL_FREE = ["isic2020", "isic2019", "isic2018", "pad_ufes20", "bcn20000"]
ALL_KAGGLE = ["ham10000", "sd198"]
ALL_REGISTRATION = ["fitzpatrick17k", "ddi", "derm7pt"]

DISPATCH = {
    "isic2016":       lambda d: download_isic_challenge("2016", d),
    "isic2017":       lambda d: download_isic_challenge("2017", d),
    "isic2018":       lambda d: download_isic_challenge("2018", d),
    "isic2019":       lambda d: download_isic_challenge("2019", d),
    "isic2020":       lambda d: download_isic_challenge("2020", d),
    "isic_api":       lambda d: download_isic_via_api(d),
    "pad_ufes20":     download_pad_ufes20,
    "bcn20000":       download_bcn20000,
    "ham10000":       download_ham10000,
    "sd198":          download_sd198,
    "fitzpatrick17k": download_fitzpatrick17k,
    "ddi":            download_ddi,
    "derm7pt":        download_derm7pt,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download public skin-cancer datasets",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=list(DISPATCH.keys()) + ["all_free", "all_kaggle", "all_registration", "list"],
        default="list",
        help=(
            "Dataset to download.\n"
            "  all_free        — ISIC 2018/2019/2020 + PAD-UFES-20 + BCN20000\n"
            "  all_kaggle      — HAM10000 + SD-198 (need Kaggle token)\n"
            "  all_registration— Fitzpatrick17k + DDI + Derm7pt (print instructions)\n"
            "  list            — show comparison table (default)"
        ),
    )
    parser.add_argument(
        "--out_dir", "-o",
        default="./data",
        help="Root output directory (default: ./data)",
    )
    parser.add_argument(
        "--isic_api_limit",
        type=int,
        default=1000,
        help="Max images to pull from ISIC API (default: 1000)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "list":
        print_summary()
        return

    if args.dataset == "all_free":
        for ds in ALL_FREE:
            DISPATCH[ds](out_dir)
        return

    if args.dataset == "all_kaggle":
        for ds in ALL_KAGGLE:
            DISPATCH[ds](out_dir)
        return

    if args.dataset == "all_registration":
        for ds in ALL_REGISTRATION:
            DISPATCH[ds](out_dir)
        return

    # Handle isic_api with limit parameter
    if args.dataset == "isic_api":
        download_isic_via_api(out_dir, limit=args.isic_api_limit)
        return

    DISPATCH[args.dataset](out_dir)


if __name__ == "__main__":
    main()
