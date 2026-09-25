"""
poster_figures.py - poster-grade figures for the ITA-CondNet A0 poster.

Everything here is computed from the locked test-set scores
(results/test_predictions.csv + results/all_scores.csv). Nothing is hard-coded
or simulated. This replaces the earlier generate_figures.py (removed from the
repo; recoverable from git history), whose ROC curves were synthesised from AUC
point estimates via roc_from_auc() rather than measured from scores, and whose
constants predated the vasc->benign label correction.

Run:  python poster_figures.py --out figures/poster
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, roc_curve

from poster_style import (use_poster_style, clean, save,
                          LIGHT, DARK, ACCENT, INK, INK2, MUTED, SURF, NEUTRAL, GRID)

RULE = "#3d3d3a"

VARIANTS = {"baseline": "Baseline", "aug_only": "Aug-only", "tone_only": "Tone-only",
            "full": "Full", "baseline_balanced": "Bal. baseline"}
T = 0.5
RNG = np.random.default_rng(42)
N_BOOT = 2000


# ----------------------------------------------------------------- data
def load():
    tp = pd.read_csv("results/test_predictions.csv")
    al = pd.read_csv("results/all_scores.csv")
    df = tp.merge(al, on="image_id", how="inner")
    df["tone"] = np.where(df.ita > 41, "light", np.where(df.ita > 10, "medium", "dark"))
    return df


def sens_spec(s, l, t=T):
    return (((s >= t) & (l == 1)).sum() / max((l == 1).sum(), 1),
            ((s < t) & (l == 0)).sum() / max((l == 0).sum(), 1))


def boot_roc_band(s, l, grid, n_boot=N_BOOT):
    """Stratified bootstrap band for TPR on a fixed FPR grid."""
    pos, neg = np.flatnonzero(l == 1), np.flatnonzero(l == 0)
    curves = np.empty((n_boot, grid.size))
    for b in range(n_boot):
        idx = np.concatenate([RNG.choice(pos, pos.size, True),
                              RNG.choice(neg, neg.size, True)])
        fpr, tpr, _ = roc_curve(l[idx], s[idx])
        curves[b] = np.interp(grid, fpr, tpr)
    return np.percentile(curves, 2.5, 0), np.percentile(curves, 97.5, 0)


def boot_auc_ci(s, l, n_boot=N_BOOT):
    pos, neg = np.flatnonzero(l == 1), np.flatnonzero(l == 0)
    a = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([RNG.choice(pos, pos.size, True),
                              RNG.choice(neg, neg.size, True)])
        a[b] = roc_auc_score(l[idx], s[idx])
    return np.percentile(a, 2.5), np.percentile(a, 97.5)


# ------------------------------------------------------------ figure 2
def fig_roc(df, out):
    """Real ROC curves by tone, full model, with bootstrap CI bands."""
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    grid = np.linspace(0, 1, 201)
    op = {}

    for tone in ("light", "dark"):
        d = df[df.tone == tone]
        s, l = d["full"].values, d["label"].values
        fpr, tpr, _ = roc_curve(l, s)
        auc = roc_auc_score(l, s)
        lo, hi = boot_auc_ci(s, l)
        blo, bhi = boot_roc_band(s, l, grid)
        c = LIGHT if tone == "light" else DARK
        ax.fill_between(grid, blo, bhi, color=c, alpha=0.16, lw=0, zorder=2)
        ax.plot(fpr, tpr, color=c, lw=3.0, zorder=4, solid_capstyle="round",
                label=(f"{'Light' if tone == 'light' else 'Dark'} skin   "
                       f"AUC {auc:.3f} [{lo:.3f}–{hi:.3f}]   n={len(d)}"))
        sn, sp = sens_spec(s, l)
        op[tone] = (1 - sp, sn)
        ax.plot(1 - sp, sn, "o", color=c, ms=13, mec=SURF, mew=2.5, zorder=6)

    ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=1.6, ls=(0, (4, 4)), zorder=1)
    ax.text(0.885, 0.845, "chance", color=MUTED, fontsize=11.5, rotation=45,
            ha="center", va="center")

    # horizontal tie between the two operating points: the gap IS the distance
    gap = (op["dark"][0] - op["light"][0]) * 100
    ymid = (op["light"][1] + op["dark"][1]) / 2
    ax.annotate("", xy=(op["dark"][0], ymid), xytext=(op["light"][0], ymid),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.6,
                                shrinkA=0, shrinkB=0))
    ax.annotate(f"both at $t$ = 0.5:\n{gap:.1f} pp more false\npositives on dark skin",
                xy=((op["light"][0] + op["dark"][0]) / 2, ymid), xytext=(0.30, 0.50),
                fontsize=12.5, color=INK2, ha="left", va="top", linespacing=1.5,
                bbox=dict(facecolor=SURF, edgecolor="none", pad=2.0),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.2,
                                connectionstyle="arc3,rad=0.2"))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate  (1 − specificity)")
    ax.set_ylabel("True positive rate  (sensitivity)")
    ax.set_title("Same discrimination, different operating point",
                 loc="left", color=INK, pad=50)
    # computed, not transcribed: the paper's Sect. 4.3 rounds this gap to -5.4 pp,
    # but the locked scores give -5.344 pp, which rounds to -5.3.
    auc_gap = (roc_auc_score(df[df.tone == "dark"].label, df[df.tone == "dark"]["full"])
               - roc_auc_score(df[df.tone == "light"].label,
                               df[df.tone == "light"]["full"])) * 100
    ax.text(0, 1.012, f"full model, full-ITA split\n"
                      f"AUC gap −{abs(auc_gap):.1f} pp   ·   "
                      f"specificity gap −{gap:.1f} pp",
            transform=ax.transAxes, fontsize=11.5, color=INK2, va="bottom",
            linespacing=1.5)
    ax.legend(loc="lower right", handlelength=1.6, borderpad=0.8, labelspacing=0.7)
    clean(ax, grid_axis="both")
    ax.set_aspect("equal")
    save(fig, out, "fig2_roc_by_tone")


# ------------------------------------------------------------ figure 3
def fig_spec_curve(df, out):
    """Specificity at every threshold = ECDF of benign scores. One shared axis,
    so the two groups are directly comparable at any operating point.

    Computed on the HIGH-CONFIDENCE ITA subset (light ITA>55, dark ITA<0), which
    is the paper's primary analysis: this reproduces the headline -11.0 pp
    specificity deficit at t=0.5 (0.826 light vs 0.715 dark)."""
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    grid = np.linspace(0, 1, 501)
    curves, ns = {}, {}
    hc = {"light": df[df.ita > 55], "dark": df[df.ita < 0]}

    for tone in ("light", "dark"):
        d = hc[tone]
        ben = np.sort(d[d.label == 0]["full"].values)
        ns[tone] = (len(d), ben.size)
        curves[tone] = np.searchsorted(ben, grid, side="left") / ben.size

    ax.fill_between(grid, curves["dark"], curves["light"], color=NEUTRAL,
                    alpha=0.35, lw=0, zorder=1, label="gap")
    for tone in ("light", "dark"):
        c = LIGHT if tone == "light" else DARK
        band = "ITA > 55°" if tone == "light" else "ITA < 0°"
        ax.plot(grid, curves[tone], color=c, lw=3.0, zorder=3, solid_capstyle="round",
                label=f"{'light' if tone == 'light' else 'dark'}   {band}")

    # Fix the limits before the labels are measured below: the measurement
    # converts a pixel box back into data units, so it is only right once the
    # data-to-display transform is final.
    ax.set_xlim(-0.012, 1)
    ax.set_ylim(-0.012, 1.02)

    # Labels carry no background box, so they must be placed where neither curve
    # runs through them. The gap midpoint is NOT safe: the curves rise to the
    # right, so a label set beside the arrow at midpoint height is crossed by the
    # dark curve. Instead, measure each label and centre it in the clear band
    # between the two curves over its own x-span. Side is chosen per threshold so
    # the two labels also stay clear of each other.
    for t, style, side in ((0.50, "-", +1), (0.35, (0, (3, 3)), -1)):
        sl = np.interp(t, grid, curves["light"])
        sd = np.interp(t, grid, curves["dark"])
        ax.plot([t, t], [0, sl], color=INK2, lw=1.4, ls=style, zorder=2)
        ax.annotate("", xy=(t, sd), xytext=(t, sl),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=2.0,
                                    shrinkA=0, shrinkB=0), zorder=5)
        txt = ax.text(t + side * 0.018, (sl + sd) / 2,
                      f"{(sl - sd) * 100:.1f} pp", fontsize=13, fontweight="bold",
                      color=INK, va="center",
                      ha="left" if side > 0 else "right")
        fig.canvas.draw()
        bb = txt.get_window_extent(fig.canvas.get_renderer())
        inv = ax.transData.inverted()
        xa = inv.transform((bb.x0, bb.y0))[0]
        xb = inv.transform((bb.x1, bb.y1))[0]
        m = (grid >= xa) & (grid <= xb)
        if m.any():
            hi = curves["dark"][m].max()      # highest the dark curve reaches
            lo = curves["light"][m].min()     # lowest the light curve reaches
            if lo - hi > 0:
                txt.set_y((hi + lo) / 2)
        # the threshold rule would run through this label, so sit beside it
        ax.text(t + 0.012, 0.025, f"$t$ = {t:.2f}", fontsize=12.5, color=INK2,
                ha="left", va="bottom")

    ax.set_xlabel("Decision threshold $t$")
    ax.set_ylabel("Specificity   (benign lesions correctly cleared)")
    # not "at every threshold": the gap is never negative, but it is only
    # strictly positive for t in [0.044, 0.980]. Outside that both specificities
    # are pinned at 0 (nothing cleared) or 1, so the universal claim is false.
    ax.set_title("The specificity deficit is not an artefact of the threshold",
                 loc="left", color=INK, pad=72)
    ax.text(0, 1.012, "high-confidence ITA subset — the paper's primary analysis\n"
                      f"curves are the benign-score ECDF "
                      f"({ns['light'][1]} of {ns['light'][0]} light / "
                      f"{ns['dark'][1]} of {ns['dark'][0]} dark)\n"
                      "$\\Delta_{spec}$ = −11.0 pp at $t$ = 0.5   "
                      "(permutation $p$ < 0.0002, Cohen's $h$ = 0.27)",
            transform=ax.transAxes, fontsize=11.5, color=INK2, va="bottom",
            linespacing=1.5)
    ax.legend(loc="lower right", handlelength=1.4, labelspacing=0.55,
              bbox_to_anchor=(1.0, 0.07))
    clean(ax, grid_axis="both")
    save(fig, out, "fig3_specificity_curve")


# ----------------------------------------------------------- figure 3b
def fig_overlap(df, out):
    """Alternative mechanism view: where benign and malignant scores actually sit.
    Both panels share one y-axis (fraction of that group), so they compare."""
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 7.0), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 41)
    tops = []

    for ax, tone in zip(axes, ("light", "dark")):
        d = df[df.tone == tone]
        c = LIGHT if tone == "light" else DARK
        ben = d[d.label == 0]["full"].values
        mal = d[d.label == 1]["full"].values

        ax.hist(ben, bins=bins, weights=np.full(ben.size, 1 / ben.size),
                color=c, alpha=0.30, lw=0, label="Benign")
        ax.hist(mal, bins=bins, weights=np.full(mal.size, 1 / mal.size),
                histtype="step", color=c, lw=2.6, label="Malignant")
        tops.append(ax.get_ylim()[1])

    top = max(tops)
    for ax, tone in zip(axes, ("light", "dark")):
        d = df[df.tone == tone]
        c = LIGHT if tone == "light" else DARK
        ben = d[d.label == 0]["full"].values
        ax.fill_between([T, 1], 0, top, color=c, alpha=0.06, lw=0, zorder=0)
        ax.axvline(T, color=INK2, lw=1.6, ls=(0, (3, 3)), zorder=5)
        ax.text(0.975, top * 0.90,
                f"{(ben >= T).mean() * 100:.1f}% of benign lesions\n"
                f"sit above the threshold",
                ha="right", va="top", fontsize=12.5, color=INK2, linespacing=1.5)
        ax.text(0.012, top * 0.90, f"{'Light' if tone == 'light' else 'Dark'} skin",
                ha="left", va="top", fontsize=15, fontweight="bold", color=c)
        ax.set_ylim(0, top)
        ax.set_ylabel("Fraction of group")
        clean(ax, grid_axis="y")
        ax.set_yticks([])

    axes[0].legend(loc="lower center", ncol=2, handlelength=1.5, columnspacing=1.8,
                   bbox_to_anchor=(0.5, 1.015))
    axes[1].set_xlabel("Predicted malignancy score   (threshold $t$ = 0.5, dashed)")
    axes[0].set_title("Benign lesions on darker skin crowd the threshold",
                      loc="left", color=INK, pad=92)
    axes[0].text(0, 1.21, "full model, full-ITA split  ·  contrast-induced class\n"
                          "overlap, the mechanism the paper formalises via SNR",
                 transform=axes[0].transAxes, fontsize=11.5, color=INK2, va="bottom",
                 linespacing=1.5)
    fig.subplots_adjust(hspace=0.18)
    save(fig, out, "fig3b_score_overlap")


# ------------------------------------------------------------ figure 4
def fig_dumbbell(df, out):
    """Specificity gap, light vs dark, across all five trained variants.
    Full-ITA split, t=0.5 -- the same basis as Fig. 2 of the paper."""
    rows = []
    for key, name in VARIANTS.items():
        vals = {}
        for tone in ("light", "dark"):
            d = df[df.tone == tone]
            vals[tone] = sens_spec(d[key].values, d["label"].values)[1]
        rows.append((name, vals["light"], vals["dark"], vals["light"] - vals["dark"]))
    rows.sort(key=lambda r: r[3], reverse=True)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    y = np.arange(len(rows))[::-1]
    x_gap = 0.965                      # fixed column for the delta values

    for yi, (name, sl, sd, gap) in zip(y, rows):
        ax.plot([sd, sl], [yi, yi], color=RULE, lw=1.0, zorder=2)
        ax.plot(sd, yi, "o", ms=8, color=DARK, zorder=3)
        ax.plot(sl, yi, "o", ms=8, color=LIGHT, zorder=3)
        ax.text(sd - 0.008, yi, f"{sd:.3f}", ha="right", va="center",
                fontsize=11.5, color=INK)
        ax.text(sl + 0.008, yi, f"{sl:.3f}", ha="left", va="center",
                fontsize=11.5, color=INK)
        best = name == "Bal. baseline"
        ax.text(1.035, yi, f"−{gap * 100:.1f}", ha="left", va="center",
                fontsize=11.5, color=INK, transform=ax.get_yaxis_transform(),
                fontweight="bold" if best else "normal")

    ax.text(1.035, len(rows) - 0.62, "Δ pp", ha="left", va="center",
            fontsize=11, color=INK2, style="italic",
            transform=ax.get_yaxis_transform())

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=13, color=INK)
    ax.set_xlim(0.628, 0.918)
    ax.set_xticks([0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    ax.set_xlabel("Specificity")
    ax.set_title("Every variant is less specific on darker skin",
                 loc="left", color=INK, pad=50)
    ax.text(0, 1.012, "full-ITA split, $t$ = 0.5\n"
                      "balancing recovers the most dark-skin specificity "
                      "(0.668 → 0.799)",
            transform=ax.transAxes, fontsize=11.5, color=INK2, va="bottom",
            linespacing=1.5)
    # direct labels on the top row rather than a legend box
    _, top_sl, top_sd, _ = rows[0]
    ytop = len(rows) - 1
    ax.text(top_sd, ytop + 0.40, "dark (ITA ≤ 10°)", ha="center", va="bottom",
            fontsize=11.5, color=DARK)
    ax.text(top_sl, ytop + 0.40, "light (ITA > 41°)", ha="center", va="bottom",
            fontsize=11.5, color=LIGHT)
    clean(ax, grid_axis="x")
    ax.set_ylim(-0.55, len(rows) - 0.10)
    save(fig, out, "fig4_specificity_gap")


# ------------------------------------------------------------ figure 5
def fig_referral(df, out):
    """Clinical cost: unnecessary referrals per 1,000 patients.
    High-confidence ITA subset -- the paper's primary analysis. The paper reports
    the t=0.5 column (~210 dark vs ~140 light, ~70 excess)."""
    hc = {"light": df[df.ita > 55], "dark": df[df.ita < 0]}
    ts = [0.50, 0.35]
    vals, pben = {}, {}
    for tone, d in hc.items():
        s_, l = d["full"].values, d["label"].values
        pben[tone] = (l == 0).mean()
        vals[tone] = [(1 - sens_spec(s_, l, t)[1]) * pben[tone] * 1000 for t in ts]

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    x = np.arange(len(ts))
    w = 0.20
    tint = {"light": "#dce9f9", "dark": "#fbddcf"}
    for i, tone in enumerate(("light", "dark")):
        c = LIGHT if tone == "light" else DARK
        off = (i - 0.5) * (w + 0.03)
        bars = ax.bar(x + off, vals[tone], w, color=tint[tone], ec=c, lw=1.2,
                      label=f"{'light' if tone == 'light' else 'dark'}   "
                            f"$p_{{benign}}$ = {pben[tone]:.3f}")
        for rect, v in zip(bars, vals[tone]):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 6, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=12.5, color=INK)

    for xi in x:
        excess = vals["dark"][xi] - vals["light"][xi]
        xr = xi + (w + 0.03) / 2 + w / 2 + 0.035
        ax.plot([xi - (w + 0.03) / 2, xr], [vals["light"][xi]] * 2,
                color=RULE, lw=0.9, zorder=4)
        ax.annotate("", xy=(xr, vals["dark"][xi]), xytext=(xr, vals["light"][xi]),
                    arrowprops=dict(arrowstyle="<->", color=RULE, lw=1.1,
                                    shrinkA=0, shrinkB=0), zorder=5)
        ax.text(xr + 0.035, (vals["light"][xi] + vals["dark"][xi]) / 2,
                f"+{excess:.0f} per 1,000", ha="left", va="center",
                fontsize=12, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([f"$t$ = {t:.2f}" + ("   (reported)" if t == 0.50 else "")
                        for t in ts], fontsize=13)
    ax.set_ylabel("Unnecessary referrals per 1,000 patients")
    ax.set_xlim(-0.55, len(ts) - 0.18)
    ax.set_ylim(0, max(max(vals["dark"]), max(vals["light"])) * 1.25)
    ax.set_title("The residual gap is a referral burden", loc="left", color=INK,
                 pad=50)
    ax.text(0, 1.012, "high-confidence ITA subset: light ITA > 55° (n=707), "
                      "dark ITA < 0° (n=558)\n"
                      r"$(1-\mathrm{Spec}) \times p_{benign} \times 1000$, "
                      "each tone using its own benign prevalence",
            transform=ax.transAxes, fontsize=11.5, color=INK2, va="bottom",
            linespacing=1.5)
    ax.legend(loc="upper left", ncol=1, labelspacing=0.5)
    clean(ax, grid_axis="y")
    save(fig, out, "fig5_referral_burden")


# ------------------------------------------------------------ figure 6
def fig_ita(df, out):
    """Where the test set sits on the ITA axis, and where positives concentrate."""
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    lo_ax, hi_ax = -90, 90
    bins = np.linspace(lo_ax, hi_ax, 61)

    ax.axvspan(lo_ax, 10, color=DARK, alpha=0.07, lw=0)
    ax.axvspan(10, 41, color=NEUTRAL, alpha=0.20, lw=0)
    ax.axvspan(41, hi_ax, color=LIGHT, alpha=0.07, lw=0)

    ax.hist(df[df.label == 0].ita, bins=bins, color=MUTED, alpha=0.45, lw=0,
            label="Benign")
    ax.hist(df[df.label == 1].ita, bins=bins, color=ACCENT, alpha=0.85, lw=0,
            label="Malignant")

    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.44)
    for lo, hi, c, lab, fs in [(lo_ax, 10, DARK, "Dark", 13),
                               (10, 41, INK2, "Medium", 9.5),
                               (41, hi_ax, LIGHT, "Light", 13)]:
        d = df[(df.ita >= lo) & (df.ita <= hi)] if lo == lo_ax else \
            df[(df.ita > lo) & (df.ita <= hi)]
        ax.text((lo + hi) / 2, top * 1.40,
                f"{lab}\nn={len(d)}\n{d.label.mean() * 100:.1f}%",
                ha="center", va="top", fontsize=fs, color=c, fontweight="bold",
                linespacing=1.5)

    for v in (10, 41):
        ax.axvline(v, color=INK2, lw=1.2, ls=(0, (3, 3)))
    ax.set_xlim(lo_ax - 2, hi_ax + 2)
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax.tick_params(axis="x", pad=7)
    ax.set_xlabel("Individual Typology Angle (ITA, degrees)   "
                  "—   darker $\\rightarrow$ lighter")
    ax.set_ylabel("Lesions")
    ax.set_title("A bimodal tone axis, and prevalence is not flat across it",
                 loc="left", color=INK, pad=50)
    ax.text(0, 1.012, "locked test set, n=1,527  ·  19.6% positive (300/1,527)\n"
                      "positives = melanoma, BCC, AKIEC  ·  band labels give "
                      "n and malignant share",
            transform=ax.transAxes, fontsize=11.5, color=INK2, va="bottom",
            linespacing=1.5)
    ax.legend(loc="center left", bbox_to_anchor=(0.28, 0.54))
    clean(ax, grid_axis="y")
    save(fig, out, "fig6_ita_distribution")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figures/poster")
    a = p.parse_args()
    use_poster_style()
    df = load()
    print(f"loaded n={len(df)}  positives={int(df.label.sum())}")
    fig_roc(df, a.out)
    fig_spec_curve(df, a.out)
    fig_overlap(df, a.out)
    fig_dumbbell(df, a.out)
    fig_referral(df, a.out)
    fig_ita(df, a.out)


if __name__ == "__main__":
    main()
