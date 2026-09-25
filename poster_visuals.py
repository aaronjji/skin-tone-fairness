"""
poster_visuals.py - the two *pictorial* figures for the A0 poster: the method
schematic and a qualitative panel of real lesions placed at their predicted score.

These are the figures that give a poster something to look at from across the
room. Both use real HAM10000 test images and the locked test-set scores.

Run:  python poster_visuals.py --out figures/poster
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from poster_style import (use_poster_style, clean, save,
                          LIGHT, DARK, ACCENT, INK, INK2, MUTED, SURF, NEUTRAL, GRID)

HAM_DIRS = [Path("data/ham10000/HAM10000_images_part_1"),
            Path("data/ham10000/HAM10000_images_part_2")]
T = 0.5


def load():
    tp = pd.read_csv("results/test_predictions.csv")
    al = pd.read_csv("results/all_scores.csv")
    df = tp.merge(al, on="image_id", how="inner")
    df["tone"] = np.where(df.ita > 41, "light", np.where(df.ita > 10, "medium", "dark"))
    return df


def img_path(image_id):
    for d in HAM_DIRS:
        p = d / f"{image_id}.jpg"
        if p.exists():
            return p
    return None


def load_thumb(image_id, size=150, square=True):
    p = img_path(image_id)
    if p is None:
        return None
    im = Image.open(p).convert("RGB")
    if square:                      # centre-crop to square, like the val transform
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return im.resize((size, size), Image.LANCZOS)


RULE = "#3d3d3a"          # hairline stroke, near-black
FILL = "#ffffff"
SOFT = "#f2f1ee"          # neutral block fill


def _background_panel(image_id):
    """The mask and statistics compute_ita() actually produced, plus a cropped
    copy of both for display.

    The statistics MUST come from the full image, exactly as compute_ita() does.
    Measuring them on the cropped-and-resized display copy instead changes the
    Otsu threshold and the pixels it selects: for ISIC_0029317 the mask goes from
    87.2% to 83.1% coverage and L* from 77.36 to 77.23, which shows as 77.4 vs
    77.2 on the figure. So compute on the full frame, then crop the image *and
    its mask* purely for display.
    """
    from skintone import _otsu, _rgb_to_lab        # the project's own code

    p = img_path(image_id)
    im = Image.open(p).convert("RGB")
    arr_full = np.array(im)

    gray = np.mean(arr_full, axis=2).astype(np.uint8)
    bg_full = gray > _otsu(gray)
    if bg_full.sum() < 100:
        bg_full = np.ones_like(gray, dtype=bool)

    lab = _rgb_to_lab(arr_full)
    L_mean = float(lab[:, :, 0][bg_full].mean())
    b_mean = float(lab[:, :, 2][bg_full].mean())
    ita = float(np.degrees(np.arctan((L_mean - 50.0) / b_mean)))

    # display only: centre-crop to square and resize, applying the SAME crop to
    # the pipeline's mask so the panel shows the mask the numbers came from
    w, h = im.size
    s = min(w, h)
    box = ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
    arr = np.array(im.crop(box).resize((320, 320), Image.LANCZOS))
    bg = np.array(Image.fromarray((bg_full * 255).astype(np.uint8))
                  .crop(box).resize((320, 320), Image.NEAREST)) > 127

    # excluded (lesion) pixels fade back; retained background stays as-is
    shown = arr.astype(float)
    shown[~bg] = shown[~bg] * 0.25 + 255 * 0.75
    return arr, shown.astype(np.uint8), L_mean, b_mean, ita


# ------------------------------------------------------------- figure 1
def fig_pipeline(df, out):
    """Method schematic, drawn in the idiom of a paper figure: hairline blocks,
    near-black type, colour reserved for the tone branch, and real intermediate
    images produced by the project's own ITA code."""
    # Type scale for this figure. Keep at 1.0: the arrow endpoints and block
    # edges below are hand-tuned absolute coordinates that do NOT scale with it,
    # so raising FS grows the centred equation into the arrow on its left and
    # the encoding block on its right. To enlarge a label, raise that label's
    # own fontsize (as the config lines at the bottom do) and re-run
    # figure_audit.py, which checks strokes and borders as well as text boxes.
    FS = 1.0
    fig, ax = plt.subplots(figsize=(16.6, 7.6))
    ax.set_xlim(0, 104.5)
    ax.set_ylim(0, 46)
    ax.set_aspect("auto")
    ax.axis("off")

    def block(x, y, w, h, fc=FILL, ec=RULE, lw=1.1, z=3):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z))

    def arrow(x1, y1, x2, y2, color=RULE, lw=1.2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, color=color, lw=lw,
                                     shrinkA=0, shrinkB=0, zorder=5,
                                     joinstyle="miter"))

    def cap(x, y, s, fs=11, color=INK, weight="normal", ha="center", va="top",
            style="normal"):
        ax.text(x, y, s, fontsize=fs * FS, color=color, fontweight=weight,
                ha=ha, va=va, zorder=6, linespacing=1.45, style=style)

    # an in-domain example (b* > 0), so the arithmetic shown is the intended case
    cand = df[(df.tone == "light") & (df.label == 0)].sort_values("full")
    sample = cand.iloc[len(cand) // 2]
    raw, masked, L_mean, b_mean, ita = _background_panel(sample.image_id)

    # ---- header ----------------------------------------------------------
    ax.text(2.0, 45.6, "ITA-CondNet", fontsize=15 * FS, fontweight="bold",
            color=INK, va="top")
    ax.text(16.5, 45.5, "EfficientNet-B2 with an ITA tone branch concatenated "
            "at the head", fontsize=13 * FS, color=INK2, va="top")
    ax.plot([2.0, 103.0], [42.8, 42.8], color=GRID, lw=1.0, zorder=1)

    # ---- input -----------------------------------------------------------
    ax.imshow(raw, extent=(2.5, 14.5, 25.0, 37.0), zorder=3, aspect="auto")
    block(2.5, 25.0, 12.0, 12.0, fc="none", z=4)
    # name the dataset in the heading: HAM10000 is distributed through the ISIC
    # archive, so its files carry ISIC_ ids and the bare id reads as "ISIC" to a
    # reviewer checking provenance -- on a paper about dataset composition that is
    # the wrong question to invite.
    cap(8.5, 40.6, "HAM10000 input image", 11.5, INK, va="bottom")
    cap(8.5, 38.2, f"{sample.image_id}   ·   224×224", 10, MUTED, va="bottom",
        style="italic")

    # ---- appearance path -------------------------------------------------
    arrow(15.6, 31.0, 21.5, 31.0)
    xs = [22.0, 25.4, 28.4, 31.0, 33.2, 35.0]
    hs = [12.0, 10.4, 8.8, 7.3, 6.0, 4.9]
    ws = [2.7, 2.4, 2.1, 1.8, 1.6, 1.4]
    greys = ["#e8e7e3", "#dedcd7", "#d3d1cb", "#c8c6bf", "#bdbbb3", "#b2b0a7"]
    for x, h, w, g in zip(xs, hs, ws, greys):
        block(x, 31.0 - h / 2, w, h, fc=g, lw=0.9)
    cap(29.0, 40.6, "EfficientNet-B2", 11.5, INK, va="bottom")
    cap(29.0, 38.2, "ImageNet-pretrained", 10, MUTED, va="bottom", style="italic")
    arrow(37.0, 31.0, 75.8, 31.0)
    cap(56.0, 32.4, "1408-d image features", 10.5, INK2, va="bottom", style="italic")

    # ---- tone path (the contribution) ------------------------------------
    ax.add_patch(Rectangle((19.5, 6.0), 57.5, 17.5, fc="#fdf7f4", ec=DARK,
                           lw=1.1, ls=(0, (5, 3)), zorder=1))
    cap(20.8, 22.7, "tone conditioning branch", 11, DARK, "bold", ha="left")

    ax.plot([2.8, 2.8], [24.2, 15.5], color=DARK, lw=1.2, zorder=5)
    arrow(2.8, 15.5, 21.4, 15.5, color=DARK)

    ax.imshow(masked, extent=(21.8, 31.8, 10.5, 20.5), zorder=3, aspect="auto")
    block(21.8, 10.5, 10.0, 10.0, fc="none", z=4)
    cap(26.8, 9.6, "background skin\n(Otsu-masked)", 10.5, INK)

    arrow(32.6, 15.5, 35.3, 15.5, color=DARK)   # equation starts at 35.86
    cap(45.5, 20.2, r"$\overline{L^{*}}$" f" = {L_mean:.1f}          "
                    r"$\overline{b^{*}}$" f" = {b_mean:.1f}", 12, INK, va="center")
    cap(45.5, 15.5, r"ITA $=\arctan\!\left(\dfrac{L^{*}-50}{b^{*}}\right)\!\cdot\!"
                    r"\dfrac{180}{\pi}$", 12.5, INK, va="center")
    cap(45.5, 11.2, f"= {ita:.1f}°", 12, INK, "bold", va="center")

    arrow(55.8, 15.5, 57.5, 15.5, color=DARK)   # equation ends at 55.14
    block(57.5, 13.3, 13.0, 4.4, fc=FILL)
    cap(64.0, 15.5, r"$[\sin\theta,\ \cos\theta,\ \theta/90]$", 11.5, INK,
        va="center")
    cap(64.0, 12.4, "angular encoding, continuous at ±90°", 10, MUTED,
        style="italic")

    arrow(70.9, 15.5, 72.4, 15.5, color=DARK)
    for dx, h in zip([0, 1.5, 3.0], [2.6, 4.2, 6.0]):
        block(72.6 + dx, 15.5 - h / 2, 1.1, h, fc="#f7e3d8", ec=DARK, lw=0.9)
    cap(76.6, 20.6, "Linear 3→16→32\n+ BN + ReLU", 10.5, INK, va="bottom",
        ha="right")
    ax.plot([76.8, 79.0], [15.5, 15.5], color=DARK, lw=1.2, zorder=5)
    ax.plot([79.0, 79.0], [15.5, 25.4], color=DARK, lw=1.2, zorder=5)
    arrow(79.0, 25.4, 79.0, 26.1, color=DARK)

    # ---- fusion + head ---------------------------------------------------
    block(76.0, 30.0, 6.0, 10.0, fc=SOFT)                # 1408 segment
    block(76.0, 26.3, 6.0, 3.4, fc="#f7e3d8", ec=DARK)   # 32 segment
    cap(79.0, 35.0, "1408", 10, INK2, va="center")
    cap(79.0, 28.0, "32", 10, DARK, va="center", weight="bold")
    cap(79.0, 41.4, "concat, 1440-d", 10.5, INK)

    arrow(82.6, 35.0, 86.0, 35.0)
    block(86.0, 31.0, 6.6, 8.0, fc=FILL)
    cap(89.3, 35.0, "FC", 12, INK, "bold", va="center")
    arrow(93.2, 35.0, 96.0, 35.0)
    cap(96.6, 35.9, "malignancy", 11, INK, ha="left", va="center")
    cap(96.6, 33.9, "score", 11, INK, ha="left", va="center")

    # ---- training / evaluation notes -------------------------------------
    cap(2.0, 4.4, "Training   AdamW  ·  BCEWithLogits (pos_weight 2.0)  ·  label "
                  "smoothing 0.05  ·  CosineAnnealingWarmRestarts  ·  40 epochs  ·  "
                  "8-crop TTA", 12, INK2, ha="left")
    cap(2.0, 2.4, "Five variants   Baseline  ·  Aug-only (dark aug + Mixup α=0.2 + "
                  "3× oversample)  ·  Tone-only  ·  Full  ·  Balanced Baseline "
                  "(pos_weight 1.0)", 12, INK2, ha="left")
    cap(2.0, 0.4, "Evaluation   locked test set n=1,527.  Primary analysis on the "
                  "high-confidence ITA subset: light ITA > 55° (n=707)  ·  "
                  "dark ITA < 0° (n=558)", 12, INK2, ha="left")
    save(fig, out, "fig1_pipeline")


# ------------------------------------------------------------- figure 7
def fig_qualitative(df, out, n_per_row=14):
    """Real benign lesions, sampled at even quantiles of each group's score
    distribution and laid out in score order.

    Even spacing rather than position-by-score: vertical position carries no
    information, so placing thumbnails at their literal x=score only collides
    where benign scores pile up near zero, and resolving that under a linear
    score axis needs postage-stamp thumbnails. Quantile sampling means the share
    past the threshold still equals the group's true flagged rate.

    n=14 is chosen so the thumbnails are as large as possible while the counts
    still land on exactly 2 vs 4. Note those counts are a rounding artefact of
    n=14: the underlying flagged rates are 28.3% vs 15.2%, a ratio of 1.86x, so
    the title says "nearly twice", not "twice".
    Row labels sit above each row rather than in a left gutter, which buys the
    full canvas width for the images: ~48 mm each on a full-width A0 band.
    """
    # geometry: equal x and y data units so thumbnails render square
    x0, x1 = -0.1, n_per_row + 0.1
    ylo, yhi = 0.55, 4.15
    fig_w = 16.6                     # same width as fig1, so type matches across figures
    fig_h = fig_w * (yhi - ylo) / (x1 - x0)
    half = 0.43

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(x0, x1)
    ax.set_ylim(ylo, yhi)
    ax.axis("off")

    qs = (np.arange(n_per_row) + 0.5) / n_per_row
    rows = (("light", 3.15, 3.90), ("dark", 1.45, 2.20))

    for tone, yc, ylab in rows:
        d = df[(df.tone == tone) & (df.label == 0)].sort_values("full")
        picks = d.iloc[np.clip((qs * len(d)).astype(int), 0, len(d) - 1)]
        c = LIGHT if tone == "light" else DARK
        n_flag = int((picks["full"].values >= T).sum())
        rate = (d["full"].values >= T).mean() * 100

        for i, (_, r) in enumerate(picks.iterrows()):
            thumb = load_thumb(r.image_id, 260)
            if thumb is None:
                continue
            flagged = r["full"] >= T
            ax.imshow(np.asarray(thumb),
                      extent=(i + 0.5 - half, i + 0.5 + half, yc - half, yc + half),
                      zorder=3, aspect="auto")
            ax.add_patch(Rectangle((i + 0.5 - half, yc - half), 2 * half, 2 * half,
                                   fc="none", ec=DARK if flagged else NEUTRAL,
                                   lw=3.0 if flagged else 1.0, zorder=4))
            ax.text(i + 0.5, yc - half - 0.05, f"{r['full']:.2f}", ha="center",
                    va="top", fontsize=10.5, zorder=5,
                    color=DARK if flagged else MUTED)

        # everything right of the divider was flagged
        xd = n_per_row - n_flag
        ax.plot([xd, xd], [yc - half - 0.26, yc + half + 0.04], color=INK2,
                lw=2.0, ls=(0, (4, 3)), zorder=6)

        band = "ITA > 41\u00b0" if tone == "light" else "ITA \u2264 10\u00b0"
        ax.text(0.0, ylab, f"{'Light' if tone == 'light' else 'Dark'} skin",
                fontsize=16, fontweight="bold", color=c, ha="left", va="top",
                zorder=6)
        ax.text(2.35, ylab - 0.02, f"{band}   \u00b7   {rate:.1f}% of all "
                                   f"{len(d)} benign lesions flagged",
                fontsize=12, color=INK2, ha="left", va="top", zorder=6)
        ax.text(n_per_row, ylab, f"{n_flag} of {n_per_row} flagged",
                fontsize=14, fontweight="bold", color=DARK, ha="right", va="top",
                zorder=6)

    # "nearly", not "twice": the underlying rates are 28.3% vs 15.2% = 1.86x.
    # The sampled counts land on exactly 2 vs 4, but that is a rounding artefact
    # of n=14 and must not be promoted into the claim.
    ax.set_title("Same model, same threshold: nearly twice as many benign "
                 "lesions flagged on darker skin", loc="left", color=INK, pad=40)
    ax.text(0, 1.035, "full model, full-ITA split, $t$ = 0.5   \u00b7   every lesion "
                      "shown is benign   \u00b7   score printed under each, "
                      "dashed line = threshold",
            transform=ax.transAxes, fontsize=11.5, color=INK2, va="bottom")
    ax.text(0, -0.03,
            "Tone is ITA-estimated from background skin, not a ground-truth "
            "Fitzpatrick label.  Thumbnails are sampled at even quantiles of each "
            "group's benign score\ndistribution (14 per row), so the proportion past "
            "the threshold reflects the true rate; spacing is even, so position is "
            "rank, not score.",
            transform=ax.transAxes, fontsize=10.5, color=MUTED, va="top",
            linespacing=1.6)
    save(fig, out, "fig7_qualitative_strip")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figures/poster")
    a = p.parse_args()
    use_poster_style()
    df = load()
    fig_pipeline(df, a.out)
    fig_qualitative(df, a.out)


if __name__ == "__main__":
    main()
