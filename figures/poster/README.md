# Poster figure kit — ITA-CondNet (A0, MICCAI)

Regenerate everything:

```bash
python poster_figures.py --out figures/poster   # data figures
python poster_visuals.py --out figures/poster   # schematic + image panel
python ita_diagnostic.py                        # fig8 (see the caveat below)
python figure_audit.py                          # layout check -- run after edits
python test_figure_audit.py                     # checks that the checks work
```

## Layout check

`figure_audit.py` renders every figure and measures each text element's actual
drawn box — and every stroke's real path — reporting seven defects that are easy
to miss by eye:

- **TEXT OVERLAP** — two labels sitting on top of each other
- **SPILLS BOX** — a label wider than the box or band it labels
- **PAST AXES** — annotation text running past the plot area, which is what makes
  `bbox_inches="tight"` pad the canvas and leave a wide empty margin
- **WHITESPACE** — more than 22% of the width empty on the right
- **STROKE THROUGH TEXT** — an arrow or rule crossing a label's glyphs. Tested
  against the text's *core* (inset 30% of its height), so a reference line grazing
  a value label's baseline does not count
- **CROSSES CONTAINER BORDER** — a label half-in, half-out of a container such as
  the dashed tone-branch panel
- **ARROW ENDPOINT INSIDE TEXT** — an arrow beginning or ending inside a label's
  glyphs, such as the tail of the arrow after `180/π` starting 0.54 units inside
  the equation. An annotation's own leader is excluded, since by design it starts
  at its own text

The last three were added after earlier versions of the audit passed a `fig1` that
had visible collisions in it: it compared text only against *text* and against small
label boxes, so it could not see a stroke through a glyph, and it deliberately
skipped wide containers to avoid false positives on background panels — which is
exactly what the dashed border is.

Four traps if you extend this, every one of which produced wrong results first:

- **`FancyArrowPatch.get_path()` is not usable.** It mixes the shaft in data
  coordinates with arrowhead geometry in a local frame, so its bounding box is
  meaningless — transforming it drops arrows into the canvas corner, where they
  "cross" whatever text sits there (that produced eight false hits). Use
  `_posA_posB` and pick the transform whose result lands inside the artist's own
  window extent.
- **The figure and axes background patches are not containers.** Counting them made
  every tick label outside its axes look like it was crossing a border.
- **Gridlines and tick marks are not strokes.** They are recessive chrome drawn
  behind labels by design, and matplotlib keeps *phantom* gridlines for ticks
  outside the view — one of those sat up in the title band and "struck through"
  the title.
- **A stroke must cross the label, not clip it.** A connector meeting a block, or
  a reference line grazing a value label's baseline, touches the padded text box
  without touching the glyphs. The check requires the crossing to span ≥35% of the
  label's width (or ≥60% of its height), and skips labels drawn on an opaque
  background box, which occludes whatever passes behind them.

An **arrow-flush-against-a-box** check was also tried and removed: connectors are
*meant* to touch the block they point into, and band dividers sit on their own band
edge, so it fired on correct design far more often than on defects. The endpoint
check above covers the half of that class which is unambiguous — an arrow must
never start inside *text*, even though it may touch a *box*. Arrow-to-box spacing
itself still needs an eye.

`test_figure_audit.py` pins all of this down with nine synthetic figures — an
arrow through a label must fire; one grazing, stopping short, or behind gridlines
must not — and runs in a second without loading torch. Run it if you change the
checks; two of these traps were caught by it rather than by eye.

It currently reports **0 issues across 9 figures**. Run it after any change; it
catches collisions that only appear at the final font sizes.

Each figure is written as **PDF** (vector — use this on the poster) and **PNG**
(300 dpi, for previews and slides). Fonts are embedded as TrueType subsets
(`pdf.fonttype 42` → `/FontFile2`, verified in the output; no Type 3), so print shops
will not substitute them.

## Which subset each figure uses

The paper reports two bases and the figures follow it, each stating its basis in a
subtitle so a reader at the board is never guessing:

| Basis | Definition | Figures | Paper |
|---|---|---|---|
| **High-confidence ITA** | light ITA > 55° (n=707), dark ITA < 0° (n=558) | `fig3`, `fig5` | the **primary analysis**; abstract's Δspec = −11.0 pp and ~70 excess referrals |
| **Full-ITA split** | light ITA > 41° (n=853), dark ITA ≤ 10° (n=560) | `fig2`, `fig3b`, `fig4`, `fig7` | Table 1 and the paper's own Figs. 1–2 (Δspec = −13.1 pp) |

`fig6` is the whole locked test set (n=1,527).

### Verified against the paper

Every figure value was checked against `camera_ready/FAIMI_main.tex`. These all
reproduce exactly from the locked scores:

- HC Full: light sens 0.813 / spec 0.826 (n=707); dark 0.803 / 0.715 (n=558)
- **Δspec (HC) = −11.0 pp**; Δauc (HC) = −4.2 pp; Balanced Baseline HC gap = −6.3 pp
- Full-ITA split: Δspec = −13.1 pp; dark spec 0.668 → 0.799 under balancing
- Tone-only dark: sens 0.871, spec 0.673
- Referrals/1,000 at t=0.5: light 140, dark 210, excess +70; p_benign 0.803 / 0.737
- Prevalence: overall 19.6% (300/1,527); light 17.6%, medium 2.6%, dark 26.2%

**One mismatch.** Sect. 4.3 states the full-split AUC gap as **−5.4 pp**. The locked
scores give **−5.344 pp**, which rounds to **−5.3**. `fig2` computes this value at
render time rather than transcribing it, so the figure reads −5.3. Worth correcting in
the paper, or checking whether that sentence predates the label correction.

### Two claims that were tightened

Both figures' *numbers* checked out; the wording around them did not:

- `fig3` was titled "holds at every threshold". The gap is never negative, but it is
  only strictly positive for **t ∈ [0.0435, 0.9795]** — below 0.044 both
  specificities are pinned at 0.000 (nothing is cleared), so there is no deficit
  there to speak of. Retitled "is not an artefact of the threshold". Peak gap is
  38.2 pp at t = 0.147.
- `fig7` was titled "twice as many". The sampled counts are exactly 2 vs 4, but that
  is a rounding artefact of n=14 — the underlying flagged rates are 28.3% vs 15.2%,
  a ratio of **1.86×**. Retitled "nearly twice as many". If you change `n_per_row`,
  re-check this wording as well as the counts.

`fig1`'s L\*, b\* and ITA are read from the **full frame**, exactly as `compute_ita()`
does. An earlier version measured them on the cropped-and-resized display copy, which
changes the Otsu threshold and therefore which pixels count as background: mask
coverage 83.1% instead of 87.2%, and L\* 77.23 instead of 77.36 — shown as 77.2 where
the pipeline's value is 77.4. The ITA survived that at one decimal by luck (75.4719
and 75.5432 both round to 75.5), which is exactly why it went unnoticed. The panel now
computes on the full image and crops the image *and its mask* only for display; its
ITA equals the cached pipeline value to 4 dp.

If you swap the sample image, keep that invariant: the numbers must come from the full
frame, or they are not the numbers the model was given.

Terminology: the paper names the model **ITA-CondNet** (Sect. 3.5) — not "SkinToneNet",
which appears only in `skintone.py`. The figures use ITA-CondNet.

## Provenance

Every number is computed at render time from `results/test_predictions.csv` +
`results/all_scores.csv` (n=1,527 locked test set, 8-crop TTA, corrected
`vasc → benign` labels). All values reproduce `results/metrics_report.txt` exactly.

> **Do not reuse `figures/*.pdf` from `generate_figures.py` on the poster.** That
> script synthesises its ROC curves from AUC point estimates
> (`roc_from_auc`, a parametric fit plus Gaussian noise — not real data), and its
> hard-coded constants (light AUC 0.9359, dark 0.8694, "Δ=6.7pp") predate the label
> correction. The current numbers are 0.9106 / 0.8572, Δ=5.3pp.

## The figures

| File | What it shows | Where it belongs |
|---|---|---|
| `fig1_pipeline` | Method schematic: image branch + ITA tone branch → joint head | Methods column, full width |
| `fig2_roc_by_tone` | Real ROC per tone, bootstrap CI bands, operating points marked | Results, top |
| `fig3_specificity_curve` | Specificity at *every* threshold; gap shaded, marked at t=0.5 / 0.35 | Results — **the headline figure** |
| `fig3b_score_overlap` | Score distributions, benign vs malignant, per tone (shared axis) | Alternative to fig3 if you prefer distributions |
| `fig4_specificity_gap` | Dumbbell: light vs dark specificity across all five variants | Ablation |
| `fig5_referral_burden` | Unnecessary referrals per 1,000 patients, t=0.5 and t=0.35 | Clinical impact / conclusion |
| `fig6_ita_distribution` | ITA axis with tone bands and per-band prevalence | Data / setup |
| `fig7_qualitative_strip` | Real benign lesions in score order, flagged ones ringed (2 of 14 vs 4 of 14) | Eye-catcher, full width |
| `fig8_ita_diagnostic` | **What the ITA split actually separates** (see below) | Not a poster figure — read it first |

`fig3` and `fig3b` are two takes on the same mechanism — pick one, don't show both.

## Suggested A0 layout (841 × 1189 mm, three columns)

```
┌──────────────────────── title ────────────────────────┐
│  fig1_pipeline  (full width, ~300 mm tall)            │
├───────────────┬───────────────┬───────────────────────┤
│ Background    │ Results       │ Clinical impact       │
│ fig6_ita      │ fig2_roc      │ fig5_referral         │
│               │ fig3_spec     │ fig4_gap              │
├───────────────┴───────────────┴───────────────────────┤
│  fig7_qualitative_strip  (full width)                 │
└───────────────────────────────────────────────────────┘
```

Place `fig7` low and wide — it is the figure that pulls people in from a distance,
and it reads without any statistics.

## Design system

- **Palette** (`poster_style.py`): light-skin group `#2a78d6` (blue), dark-skin group
  `#eb6834` (orange), accent `#4a3aa7` (violet). Colour means the same thing in every
  figure. Validated colourblind-safe: worst-pair ΔE 24.7 under CVD simulation, 33.6
  normal vision — well past the ≥8 / ≥15 floors.
- **Type** is sized for reading at ~1.5 m. If your poster template uses a different
  family, change `font.sans-serif` in `poster_style.py` once.
- **Surface** is pure `#ffffff`, so the figures sit flush on a white poster with no
  faint warm panel behind them. The palette was re-validated against this surface:
  all six checks still pass (worst-pair CVD ΔE 24.7, contrast ≥ 3:1).
- Identity is never carried by colour alone — every series is also direct-labelled.

## Scaling for print

Figures are sized in inches at roughly the proportions they should occupy. In LaTeX
(`\includegraphics[width=...]`) or PowerPoint, scale **uniformly** — do not stretch
one axis, or the type stops matching across figures.

**`use_poster_style(scale=...)` only moves rcParams-driven text** — titles, axis
labels and tick labels. It does not touch any text that passes an explicit
`fontsize=`, which is most annotations and *all* of `fig1` and `fig7`.

**Do not raise `fig1`'s `FS` above 1.0.** Its arrow endpoints and block edges are
hand-tuned absolute coordinates that do not scale with the type, and the layout has
no slack: at `FS = 1.15` the centred ITA equation grows ~1.5 units each way, running
into `arrow(32.6 … 35.6)` on its left (an 18 px strikethrough through the glyphs) and
`block(57.5, w=13.0)` on its right. Chasing it the other way means re-tuning six
coordinates between an equation ending at ~56.9 and the linear blocks starting at
72.6.

To make one label bigger, raise **that label's own fontsize** instead — the three
bottom config lines run at `12` rather than the `10.5` used elsewhere, which is the
type bump that was actually wanted. Re-run `figure_audit.py` afterwards.

## Read this before printing: the HAM10000 tone split does not separate skin tone

Building `fig7` surfaced this, and `ita_diagnostic.py` (run it — it caches to
`results/ita_lab_components.csv`) confirms it across all 1,527 test images:

```
images with L* > 50            : 99.9%
sign(ITA) == sign(b*)          : 99.9%
corr(ITA, L*)  [lightness]     : +0.050
corr(ITA, b*)  [yellow-blue]   : +0.647
mean L*:  assigned light 69.4   assigned dark 68.2   (d = -0.18)
```

ITA = arctan((L\*−50) / b\*) · 180/π is only meaningful where **b\* > 0**, which holds
for skin (typical b\* is 10–25, skin is yellowish). In this test set L\* > 50 for
99.9% of images, so the numerator is always positive and **the sign of ITA is fixed
entirely by the sign of b\***. The "dark" group is therefore the set of images whose
Otsu-selected background has a *negative b\** — a blue/magenta colour cast — not
darker skin. The two groups' mean lightness differs by 1.2 L\* units.

This is what produces the bimodal ITA axis in `fig6` (arctan saturating toward ±90°)
and why the "dark" thumbnails in `fig7` look as light as the "light" ones.

The subgroup gap itself is real and reproducible — but on this dataset it is a gap
between colour-cast groups, and the skin-tone reading of it is not supported. Options,
roughly in increasing order of work:

1. **Relabel** these figures to what the split defensibly is (e.g. "ITA < 0" vs
   "ITA > 41", or "negative-b\* subgroup"), and drop the tone claim on HAM10000.
2. **Lead with DDI instead**, which has ground-truth Fitzpatrick labels — though its
   overall AUC is 0.62 and dark-skin sensitivity 0.229, so it is weak evidence.
3. **Fix the tone estimate**: restrict ITA to b\* > 0, or replace the Otsu
   `gray > threshold` background rule (it selects the *brightest* pixels, which are
   often specular highlights and vignette rather than skin) with a proper skin
   segmentation.

The figure scripts still say "light skin"/"dark skin" because that is what the current
paper claims; change the labels once you have decided which way to go. `fig8` is the
evidence, not a poster panel.

Separately, `ddi_results/ddi_section.txt` still quotes the pre-correction "6.7pp"
HAM10000 gap; the corrected value is 5.3pp.

## Sizing fig7 on the board

`fig7` is built to be the full-width band. At 800 mm wide it renders **~283 mm tall
with ~48 mm thumbnails** — larger than they would be at any layout that positions
each lesion at its literal x = score, because benign scores bunch near zero and a
linear score axis wastes most of the canvas on the empty 0.5–1.0 stretch.

Two consequences of that choice, both stated on the figure itself: spacing is even,
so horizontal position is **rank, not score** (each lesion's actual score is printed
beneath it); and the sample is 14 per row, chosen because it is the largest count
whose flagged tallies still land on exactly **2 vs 4** — the "twice as many" the
title claims. Changing `n_per_row` changes those tallies, so re-check the title if
you do.

Don't scale fig7 narrower than about 500 mm: below that the printed scores stop
being readable at poster distance, and the thumbnails drop under 30 mm.
