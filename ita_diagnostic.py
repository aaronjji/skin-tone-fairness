"""
ita_diagnostic.py - what the ITA split is actually separating.

ITA = arctan((L* - 50) / b*) * 180/pi is defined for skin, where b* > 0 (skin is
yellowish; typical b* is 10-25). The sign of ITA therefore carries tone only while
b* stays positive. This script computes L* and b* over the same Otsu background
mask compute_ita() uses, for the whole locked test set, and plots what the
light/dark stratification is actually cutting on.

Run:  python ita_diagnostic.py            # caches results/ita_lab_components.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from poster_style import use_poster_style, clean, save, LIGHT, DARK, ACCENT, INK, INK2, MUTED, NEUTRAL, GRID, SURF

CACHE = Path("results/ita_lab_components.csv")
DIRS = [Path("data/ham10000/HAM10000_images_part_1"),
        Path("data/ham10000/HAM10000_images_part_2")]


def img_path(i):
    for d in DIRS:
        p = d / f"{i}.jpg"
        if p.exists():
            return p
    return None


def compute_components():
    from skintone import _otsu, _rgb_to_lab

    tp = pd.read_csv("results/test_predictions.csv")
    out = []
    for n, (_, r) in enumerate(tp.iterrows(), 1):
        p = img_path(r.image_id)
        if p is None:
            continue
        arr = np.array(Image.open(p).convert("RGB"))
        gray = np.mean(arr, axis=2).astype(np.uint8)
        bg = gray > _otsu(gray)
        if bg.sum() < 100:
            bg = np.ones_like(gray, dtype=bool)
        lab = _rgb_to_lab(arr)
        out.append(dict(image_id=r.image_id, ita=r.ita, label=r.label,
                        L=float(lab[:, :, 0][bg].mean()),
                        b=float(lab[:, :, 2][bg].mean())))
        if n % 200 == 0:
            print(f"  {n}/{len(tp)}", flush=True)
    df = pd.DataFrame(out)
    CACHE.parent.mkdir(exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def fig_diagnostic(df, out="figures/poster"):
    df = df.copy()
    df["tone"] = np.where(df.ita > 41, "light",
                          np.where(df.ita > 10, "medium", "dark"))
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.0),
                             gridspec_kw=dict(width_ratios=[1.25, 1]))

    # ---- (a) where the groups actually sit in L*, b* --------------------
    ax = axes[0]
    for tone, c, lab in [("light", LIGHT, "assigned light"),
                         ("medium", NEUTRAL, "medium"),
                         ("dark", DARK, "dark")]:
        d = df[df.tone == tone]
        ax.scatter(d.b, d.L, s=16, color=c, alpha=0.55, lw=0, label=lab, zorder=3)

    ax.axvline(0, color=INK, lw=1.8, zorder=4)
    ax.axhline(50, color=INK2, lw=1.4, ls=(0, (4, 3)), zorder=4)
    ax.text(0.6, 96, "$b^{*}=0$\nthe split", fontsize=12, color=INK,
            fontweight="bold", va="top", linespacing=1.5)
    ax.text(ax.get_xlim()[1], 49.2, "$L^{*}=50$  (ITA numerator changes sign)",
            fontsize=11, color=INK2, ha="right", va="top")

    frac = (df.L > 50).mean() * 100
    ax.text(0.0, -0.165, f"{frac:.1f}% of images lie above $L^{{*}}=50$, so the sign\n"
                         f"of ITA is set by the sign of $b^{{*}}$ alone, not by lightness.",
            transform=ax.transAxes, fontsize=12, color=INK, va="top",
            linespacing=1.6)

    ax.set_xlabel("$b^{*}$  (blue ←→ yellow)  over background skin")
    ax.set_ylabel("$L^{*}$  (lightness)  over background skin")
    ax.set_title("The split is a cut on $b^{*}$, not on lightness",
                 loc="left", color=INK, pad=38)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3,
              columnspacing=1.6, handletextpad=0.4)
    clean(ax, grid_axis="both")

    # ---- (b) the thing tone is supposed to mean ------------------------
    ax = axes[1]
    data = [df[df.tone == "light"].L.values, df[df.tone == "dark"].L.values]
    parts = ax.violinplot(data, positions=[0, 1], widths=0.7, showextrema=False)
    for pc, c in zip(parts["bodies"], (LIGHT, DARK)):
        pc.set_facecolor(c)
        pc.set_alpha(0.35)
        pc.set_edgecolor(c)
        pc.set_linewidth(1.5)
    for i, (d, c) in enumerate(zip(data, (LIGHT, DARK))):
        ax.plot([i - 0.17, i + 0.17], [d.mean()] * 2, color=c, lw=3,
                solid_capstyle="round", zorder=4)
        ax.text(i, d.mean() + 2.0, f"{d.mean():.1f}", ha="center", fontsize=13,
                fontweight="bold", color=c)

    pooled = np.sqrt((data[0].var() + data[1].var()) / 2)
    dcoh = (data[1].mean() - data[0].mean()) / pooled
    ax.text(0.5, -0.165, f"difference in mean $L^{{*}}$ = "
                         f"{data[1].mean()-data[0].mean():+.1f}\n"
                         f"Cohen's $d$ = {dcoh:+.2f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=12.5, color=INK,
            fontweight="bold", linespacing=1.6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["assigned\nlight", "assigned\ndark"], fontsize=13)
    ax.set_ylabel("$L^{*}$  (lightness)")
    ax.set_title("Both groups are equally light", loc="left", color=INK, pad=38)
    clean(ax, grid_axis="y")

    fig.subplots_adjust(wspace=0.22)
    save(fig, out, "fig8_ita_diagnostic")


if __name__ == "__main__":
    use_poster_style()
    if CACHE.exists():
        print(f"using cache {CACHE}")
        df = pd.read_csv(CACHE)
    else:
        print("computing L*/b* over the test set (a few minutes)...")
        df = compute_components()
    r_L = np.corrcoef(df.ita, df.L)[0, 1]
    r_b = np.corrcoef(df.ita, df.b)[0, 1]
    same = (np.sign(df.ita) == np.sign(df.b)).mean() * 100
    print(f"\nn = {len(df)}")
    print(f"  images with L* > 50            : {(df.L > 50).mean()*100:.1f}%")
    print(f"  sign(ITA) == sign(b*)          : {same:.1f}%")
    print(f"  corr(ITA, L*)  [lightness]     : {r_L:+.3f}")
    print(f"  corr(ITA, b*)  [yellow-blue]   : {r_b:+.3f}")
    fig_diagnostic(df)
