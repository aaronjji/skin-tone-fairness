"""
compute_metrics.py
==================
Comprehensive post-hoc evaluation after vasc→benign label correction.
Loads all five trained checkpoints, runs 8-crop TTA, and produces:

  3a. Full ablation table  (5 variants × {overall, light, dark}):
        AUC + 95% CI (bootstrap 2000), Sens + 95% CI, Spec, n, n_pos
  3b. HC-ITA subset        (light ITA>55, dark ITA<0) + dark−light gaps
  3c. Three pre-specified permutation tests (5000 iter, seed=42)
  3d. New refer prevalence (positive = {mel, bcc, akiec})
  3e. Referral-burden numbers at t=0.50 and t=0.35
  3f. DDI confirmation statement

Usage:
    python compute_metrics.py

Outputs:
    results/all_scores.csv       — per-image scores for all 5 variants (cached)
    results/metrics_report.txt   — paste-ready tables
"""

import json, sys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix

from skintone import SkinToneNet, encode_ita, IMG_SIZE

# ── Config ────────────────────────────────────────────────────────────────────

SEED            = 42
BOOTSTRAP_ITERS = 2000
PERM_ITERS      = 5000
T_DEFAULT       = 0.50
T_CALIB         = 0.35

MALIGNANT = {"mel", "bcc", "akiec"}   # corrected: vasc → benign

BASE      = Path(".")
DATA      = BASE / "data" / "ham10000"
META_CSV  = DATA / "HAM10000_metadata.csv"
ITA_CACHE = BASE / "results" / "ita_cache.json"
SPLIT_IDX = BASE / "results" / "split_indices.json"
RESULTS   = BASE / "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Order determines table row order
VARIANTS = [
    ("baseline",          RESULTS / "baseline_best.pt",          False),
    ("aug_only",          RESULTS / "aug_only_best.pt",          False),
    ("tone_only",         RESULTS / "tone_only_best.pt",         True),
    ("full",              RESULTS / "full_best.pt",              True),
    ("baseline_balanced", RESULTS / "baseline_balanced_best.pt", False),
]

DISPLAY = {
    "baseline":          "Baseline",
    "aug_only":          "Aug-only",
    "tone_only":         "Tone-only",
    "full":              "Full",
    "baseline_balanced": "Bal. Baseline",
}

np.random.seed(SEED)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_df():
    meta = pd.read_csv(META_CSV)
    meta.columns = [c.lower().strip() for c in meta.columns]

    with open(ITA_CACHE) as f:
        ita_cache = json.load(f)
    with open(SPLIT_IDX) as f:
        splits = json.load(f)

    test_ids = set(splits["test"])
    df = meta[meta["image_id"].isin(test_ids)].copy().reset_index(drop=True)

    df["label"] = df["dx"].str.lower().str.strip().isin(MALIGNANT).astype(int)
    df["ita"]   = pd.to_numeric(
        df["image_id"].map(lambda x: ita_cache.get(x)), errors="coerce"
    )

    def tone_group(ita):
        if pd.isna(ita):
            return "unknown"
        return "light" if ita > 41.0 else ("medium" if ita > 10.0 else "dark")

    df["tone_group"] = df["ita"].apply(tone_group)

    img_map = {}
    for d in [DATA / "HAM10000_images_part_1", DATA / "HAM10000_images_part_2"]:
        if d.exists():
            for p in d.iterdir():
                img_map[p.stem] = str(p)

    df["image_path"] = df["image_id"].map(img_map).fillna("")
    df = df[df["image_path"] != ""].copy().reset_index(drop=True)

    print(f"Test set: n={len(df)}, positives={df['label'].sum()} "
          f"({df['label'].mean()*100:.1f}%), seed=42 split")
    tc = df["tone_group"].value_counts().to_dict()
    print(f"Tone groups: light={tc.get('light',0)}, medium={tc.get('medium',0)}, "
          f"dark={tc.get('dark',0)}, unknown={tc.get('unknown',0)}")
    return df


# ── TTA inference ─────────────────────────────────────────────────────────────

def make_tta_transforms():
    tfms = []
    for rot in [0, 90, 180, 270]:
        for flip in [False, True]:
            ops = [T.Resize(256), T.CenterCrop(IMG_SIZE)]
            if rot:
                ops.append(T.RandomRotation((rot, rot)))
            if flip:
                ops.append(T.RandomHorizontalFlip(p=1.0))
            ops += [T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            tfms.append(T.Compose(ops))
    return tfms


TTA_TFMS = make_tta_transforms()


@torch.no_grad()
def run_inference(ckpt_path, use_tone, df, desc=""):
    model = SkinToneNet(use_tone=use_tone, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  TTA {desc}", leave=False):
        img     = Image.open(row["image_path"]).convert("RGB")
        ita_val = row["ita"] if not pd.isna(row["ita"]) else 0.0
        enc     = torch.tensor(encode_ita(ita_val), dtype=torch.float32).unsqueeze(0).to(DEVICE)

        ps = []
        for tfm in TTA_TFMS:
            x   = tfm(img).unsqueeze(0).to(DEVICE)
            lgt = model(x, enc) if use_tone else model(x)
            ps.append(torch.sigmoid(lgt).item())
        scores.append(float(np.mean(ps)))

    return np.array(scores)


# ── Metric primitives ─────────────────────────────────────────────────────────

def met(scores, labels, t=T_DEFAULT):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return dict(auc=float("nan"), sens=float("nan"), spec=float("nan"),
                    n=int(len(labels)), n_pos=int(labels.sum()) if len(labels) else 0)
    preds = (scores >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return dict(
        auc  = float(roc_auc_score(labels, scores)),
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        n    = int(len(labels)),
        n_pos= int(labels.sum()),
    )


def bci(scores, labels, fn, n_iter=BOOTSTRAP_ITERS, seed=SEED):
    rng  = np.random.default_rng(seed)
    vals = []
    n    = len(scores)
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        sl, sp = labels[idx], scores[idx]
        if len(np.unique(sl)) < 2:
            continue
        try:
            vals.append(fn(sp, sl))
        except Exception:
            pass
    if not vals:
        return float("nan"), float("nan"), float("nan")
    v = np.array(vals)
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def perm_test(sa, sb, labels, fn, n_iter=PERM_ITERS, seed=SEED):
    rng = np.random.default_rng(seed)
    try:
        obs = fn(sa, labels) - fn(sb, labels)
    except Exception:
        return float("nan"), float("nan")
    nulls = []
    for _ in range(n_iter):
        swap = rng.integers(0, 2, len(sa)).astype(bool)
        a2   = np.where(swap, sb, sa)
        b2   = np.where(swap, sa, sb)
        try:
            nulls.append(fn(a2, labels) - fn(b2, labels))
        except Exception:
            pass
    nulls = np.array(nulls)
    return float(obs), float(np.mean(np.abs(nulls) >= abs(obs)))


def cohen_h(p1, p2):
    return float(
        2 * np.arcsin(np.sqrt(np.clip(p1, 1e-9, 1 - 1e-9)))
        - 2 * np.arcsin(np.sqrt(np.clip(p2, 1e-9, 1 - 1e-9)))
    )


def _auc_fn(s, l):
    return roc_auc_score(l, s)


def _sens_fn(s, l, t=T_DEFAULT):
    p = (s >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(l, p, labels=[0, 1]).ravel()
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def _spec_fn(s, l, t=T_DEFAULT):
    p = (s >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(l, p, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


# ── Formatting helpers ────────────────────────────────────────────────────────

def _f(v, d=4):
    return f"{v:.{d}f}" if isinstance(v, float) and not np.isnan(v) else "  N/A"


def _fd(v, d=4):
    return f"{v:+.{d}f}" if isinstance(v, float) and not np.isnan(v) else "  N/A"


def _ci(lo, hi):
    if np.isnan(lo) or np.isnan(hi):
        return "       N/A      "
    return f"[{lo:.3f}-{hi:.3f}]"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\nDevice: {DEVICE}")
    df     = load_test_df()
    labels = df["label"].values.astype(int)
    itas   = df["ita"].values.astype(float)

    # ── Load / compute scores ─────────────────────────────────────────────────
    scores_path = RESULTS / "all_scores.csv"
    cached      = {}
    if scores_path.exists():
        sc_df = pd.read_csv(scores_path)
        for col in sc_df.columns:
            if col != "image_id":
                cached[col] = sc_df[col].values
        print(f"Cached scores loaded from {scores_path}: {list(cached.keys())}")
    else:
        sc_df = df[["image_id"]].copy()

    all_scores = {}
    for name, ckpt, use_tone in VARIANTS:
        if name in cached:
            all_scores[name] = cached[name]
            print(f"  {name}: from cache")
            continue
        if not ckpt.exists():
            print(f"  [SKIP] {name}: checkpoint not found at {ckpt}")
            continue
        print(f"\n  Inferring {name} ({ckpt.name})...")
        sc              = run_inference(ckpt, use_tone, df, desc=name)
        all_scores[name] = sc
        sc_df[name]      = sc

    sc_df.to_csv(scores_path, index=False)
    print(f"\nScores cached -> {scores_path}\n")

    # ── Subset masks ──────────────────────────────────────────────────────────
    masks = {
        "overall":  np.ones(len(df), dtype=bool),
        "light":    itas > 41.0,
        "medium":   (itas > 10.0) & (itas <= 41.0),
        "dark":     itas <= 10.0,
        "hc_light": itas > 55.0,
        "hc_dark":  itas < 0.0,
    }

    # Pre-compute all per-variant metrics (needed for permutation tests + CI)
    abl = {}   # abl[variant][tone] = metric dict with CIs
    for name, _, _ in VARIANTS:
        if name not in all_scores:
            continue
        sc      = all_scores[name]
        abl[name] = {}
        for tone in ("overall", "light", "medium", "dark"):
            mask    = masks[tone]
            ts, tl  = sc[mask], labels[mask]
            m       = met(ts, tl)
            am, al, ah = bci(ts, tl, _auc_fn)
            sm, sl, sh = bci(ts, tl, _sens_fn)
            m.update(auc_lo=al, auc_hi=ah, sens_lo=sl, sens_hi=sh)
            abl[name][tone] = m

    # ── Build report ──────────────────────────────────────────────────────────
    lines = []

    def sep(char="=", w=115):
        lines.append(char * w)

    # ── 3a. Full ablation table ───────────────────────────────────────────────
    sep()
    lines.append("TABLE 3a — FULL ABLATION  (corrected: positive = {mel, bcc, akiec}; vasc -> benign)")
    lines.append("          n=1527 locked test set, t=0.5, 8-crop TTA, 2000-iter bootstrap 95% CI")
    sep()
    lines.append(
        f"{'Variant':<16} {'Tone':<8} {'AUC':>6} {'[95% CI]':>17} "
        f"{'Sens':>6} {'[95% CI]':>17} {'Spec':>6} {'n':>5} {'n+':>5}"
    )
    sep("-")

    for name, _, _ in VARIANTS:
        if name not in abl:
            lines.append(f"{DISPLAY[name]:<16} — checkpoint not found, skipped")
            lines.append("")
            continue
        for tone in ("overall", "light", "dark"):
            m = abl[name][tone]
            lines.append(
                f"{DISPLAY[name]:<16} {tone:<8} {_f(m['auc']):>6} {_ci(m['auc_lo'],m['auc_hi']):>17} "
                f"{_f(m['sens'],3):>6} {_ci(m['sens_lo'],m['sens_hi']):>17} "
                f"{_f(m['spec'],3):>6} {masks[tone].sum():>5} {int(labels[masks[tone]].sum()):>5}"
            )
        lines.append("")

    sep()
    lines.append("")

    # ── 3b. HC-ITA subset ─────────────────────────────────────────────────────
    sep()
    lines.append("TABLE 3b — HIGH-CONFIDENCE ITA SUBSET")
    lines.append("          Light HC: ITA>55, Dark HC: ITA<0  |  point estimates only")
    sep()
    lines.append(
        f"{'Variant':<16} {'HC Tone':<14} {'AUC':>6} {'Sens':>6} {'Spec':>6} "
        f"{'n':>5} {'n+':>5}"
    )
    sep("-")

    for name, _, _ in VARIANTS:
        if name not in all_scores:
            continue
        sc = all_scores[name]
        hc = {}
        for key, lbl in (("hc_light", "light (ITA>55)"), ("hc_dark", "dark  (ITA<0)")):
            mask   = masks[key]
            ts, tl = sc[mask], labels[mask]
            m      = met(ts, tl)
            hc[key] = m
            lines.append(
                f"{DISPLAY[name]:<16} {lbl:<14} {_f(m['auc']):>6} "
                f"{_f(m['sens'],3):>6} {_f(m['spec'],3):>6} "
                f"{mask.sum():>5} {int(labels[mask].sum()):>5}"
            )
        hl = hc.get("hc_light", {}); hd = hc.get("hc_dark", {})
        d_auc  = (hd.get("auc",  float("nan")) - hl.get("auc",  float("nan")))
        d_sens = (hd.get("sens", float("nan")) - hl.get("sens", float("nan")))
        d_spec = (hd.get("spec", float("nan")) - hl.get("spec", float("nan")))
        lines.append(
            f"{DISPLAY[name]:<16} {'d dark-light':<14} "
            f"{_fd(d_auc):>6} {_fd(d_sens,3):>6} {_fd(d_spec,3):>6}"
        )
        lines.append("")

    sep()
    lines.append("")

    # ── 3c. Permutation tests ─────────────────────────────────────────────────
    sep()
    lines.append("TABLE 3c — PRE-SPECIFIED PERMUTATION TESTS  (5000 iter, seed=42)")
    lines.append("          Comparison: Full model vs Baseline on DARK subset (ITA<=10)")
    sep()
    _ch = "Cohen's h"
    lines.append(
        f"{'#  Test':<32} {'Obs. Gap':>10} {'p-value':>9} {_ch:>10}"
    )
    sep("-")

    if "full" in all_scores and "baseline" in all_scores:
        dm     = masks["dark"]
        sf     = all_scores["full"][dm]
        sb_    = all_scores["baseline"][dm]
        ld     = labels[dm]

        tests = [
            ("1. Dark spec  (Full vs Base, t=0.5)",  _spec_fn, "spec"),
            ("2. Dark sens  (Full vs Base, t=0.5)",  _sens_fn, "sens"),
            ("3. Dark AUC   (Full vs Base)       ",  _auc_fn,  "auc"),
        ]
        for desc, fn, key in tests:
            obs, p = perm_test(sf, sb_, ld, fn)
            f_val  = abl.get("full",     {}).get("dark", {}).get(key, float("nan"))
            b_val  = abl.get("baseline", {}).get("dark", {}).get(key, float("nan"))
            h      = cohen_h(f_val, b_val)
            lines.append(
                f"{desc:<32} {obs:>+10.4f} {p:>9.4f} {h:>10.4f}"
            )
    else:
        lines.append("  [SKIP] full or baseline checkpoint not available")

    sep()
    lines.append("")

    # ── 3d. Refer prevalence ──────────────────────────────────────────────────
    sep(w=65)
    lines.append("TABLE 3d — REFERRAL TRIAGE PREVALENCE (corrected labels)")
    sep(w=65)
    for tone in ("overall", "light", "dark", "medium"):
        mask  = masks[tone]
        n_t   = int(mask.sum())
        n_pos = int(labels[mask].sum())
        if n_t == 0:
            continue
        lines.append(
            f"  {tone:<8}  n={n_t:4d}  positives={n_pos:3d}  "
            f"prevalence={n_pos/n_t*100:.1f}%  "
            f"p_benign={1-n_pos/n_t:.4f}"
        )
    n_tot  = int(labels.sum())
    p_ben  = (len(labels) - n_tot) / len(labels)
    lines.append(f"\n  Overall: {n_tot} positives / {len(labels)} total  "
                 f"-> refer rate={n_tot/len(labels)*100:.1f}%  "
                 f"p_benign={p_ben:.4f}")
    sep(w=65)
    lines.append("")

    # ── 3e. Referral burden ───────────────────────────────────────────────────
    sep(w=95)
    lines.append("TABLE 3e — REFERRAL BURDEN  (full model, corrected labels)")
    lines.append(
        f"          unnecessary_per_1000 = (1 - spec) x p_benign x 1000"
        f"   [p_benign = {p_ben:.4f}]"
    )
    sep(w=95)
    lines.append(
        f"{'Tone':<8} {'t':>5}  {'Spec':>6}  {'FPR':>6}  "
        f"{'Unnecessary/1000':>18}  {'n_benign':>10}"
    )
    sep("-", w=95)

    if "full" in all_scores:
        sf = all_scores["full"]
        for tone in ("light", "dark"):
            mask = masks[tone]
            ts, tl = sf[mask], labels[mask]
            for t_val in (T_DEFAULT, T_CALIB):
                m      = met(ts, tl, t=t_val)
                spec   = m.get("spec", float("nan"))
                fpr    = 1.0 - spec if not np.isnan(spec) else float("nan")
                burden = fpr * p_ben * 1000 if not np.isnan(fpr) else float("nan")
                n_ben  = int((tl == 0).sum())
                lines.append(
                    f"{tone:<8} {t_val:>5.2f}  {_f(spec,3):>6}  {_f(fpr,3):>6}  "
                    f"{_f(burden,1):>18}  {n_ben:>10}"
                )

        spec_d5 = met(sf[masks["dark"]],  labels[masks["dark"]],  t=0.50).get("spec", float("nan"))
        spec_l5 = met(sf[masks["light"]], labels[masks["light"]], t=0.50).get("spec", float("nan"))
        spec_d3 = met(sf[masks["dark"]],  labels[masks["dark"]],  t=T_CALIB).get("spec", float("nan"))
        spec_l3 = met(sf[masks["light"]], labels[masks["light"]], t=T_CALIB).get("spec", float("nan"))

        exc_50 = ((1-spec_d5)-(1-spec_l5)) * p_ben * 1000
        exc_35 = ((1-spec_d3)-(1-spec_l3)) * p_ben * 1000
        lines.append(
            f"\n  Excess unnecessary referrals dark vs light @ t=0.50: {exc_50:+.1f} / 1 000 patients"
        )
        lines.append(
            f"  Excess unnecessary referrals dark vs light @ t=0.35: {exc_35:+.1f} / 1 000 patients"
        )
        lines.append(
            f"  Dark specificity  t=0.50 -> t=0.35: "
            f"{_f(spec_d5,3)} -> {_f(spec_d3,3)}  (d = {spec_d3-spec_d5:+.3f})"
        )
        lines.append(
            f"  Light specificity t=0.50 -> t=0.35: "
            f"{_f(spec_l5,3)} -> {_f(spec_l3,3)}  (d = {spec_l3-spec_l5:+.3f})"
        )
    else:
        lines.append("  [SKIP] full checkpoint not available")

    sep(w=95)
    lines.append("")

    # ── 3f. DDI confirmation ──────────────────────────────────────────────────
    sep(w=70)
    lines.append("TABLE 3f — DDI EXTERNAL VALIDATION STATUS")
    sep(w=70)
    lines.append("  DDI labels: read from ddi_metadata.csv 'malignant' column,")
    lines.append("  which is DDI's own biopsy-confirmed independent ground truth.")
    lines.append("  Neither run_ddi_inference.py nor evaluate_ddi.py uses any")
    lines.append("  HAM dx-to-label mapping.  DDI numbers are UNCHANGED.")
    sep(w=70)

    report = "\n".join(lines)
    print("\n" + report)

    out_path = RESULTS / "metrics_report.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved -> {out_path}")


if __name__ == "__main__":
    main()
