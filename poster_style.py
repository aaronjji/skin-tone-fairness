"""
poster_style.py — shared visual language for the SkinToneNet A0 poster figures.

Palette is validated colourblind-safe (blue<->orange: CVD dE 24.7, normal dE 33.6;
with violet accent: worst all-pairs CVD dE 13.0). Light/print surface only.
Figures are sized for an A0 poster column and exported as vector PDF + 300dpi PNG.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- palette ------------------------------------------------------------
LIGHT   = "#2a78d6"   # light-skin group (categorical slot 1)
DARK    = "#eb6834"   # dark-skin  group (categorical slot 2)
ACCENT  = "#4a3aa7"   # violet accent (chosen variant / highlight)
INK     = "#0b0b0b"
INK2    = "#52514e"
MUTED   = "#8a8a85"
GRID    = "#e3e2de"
SURF    = "#ffffff"   # flush with the poster's white, no warm panel
NEUTRAL = "#c9c8c3"

TONE_C  = {"light": LIGHT, "dark": DARK}
TONE_L  = {"light": "Light skin (ITA > 41\u00b0)", "dark": "Dark skin (ITA \u2264 10\u00b0)"}


def use_poster_style(scale=1.0):
    """Typography sized to be read from ~1.5 m on an A0 board."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size":        13 * scale,
        "axes.titlesize":   16 * scale,
        "axes.labelsize":   14 * scale,
        "legend.fontsize":  12.5 * scale,
        "xtick.labelsize":  12.5 * scale,
        "ytick.labelsize":  12.5 * scale,
        "axes.titleweight": "bold",
        "axes.titlepad":    14,
        "axes.labelcolor":  INK2,
        "text.color":       INK,
        "xtick.color":      INK2,
        "ytick.color":      INK2,
        "axes.edgecolor":   GRID,
        "axes.linewidth":   1.0,
        "grid.color":       GRID,
        "grid.linewidth":   1.0,
        "legend.frameon":   False,
        "figure.facecolor": SURF,
        "axes.facecolor":   SURF,
        "savefig.facecolor": SURF,
        "figure.dpi":       110,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "pdf.fonttype":     42,   # embed TrueType, not Type3 -> safe for print shops
        "ps.fonttype":      42,
    })


def clean(ax, grid_axis="y"):
    """Hairline recessive chrome: no top/right spines, one solid hairline grid."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=1.0, ls="-")
        ax.set_axisbelow(True)
    ax.tick_params(length=0)
    return ax


def save(fig, out_dir, name):
    from pathlib import Path
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved  {name}.pdf / .png")
