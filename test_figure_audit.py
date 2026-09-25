"""
test_figure_audit.py - regression test for the layout checks themselves.

The checks are only useful if they fire on real collisions and stay quiet on
correct design. Both failure modes have happened here: the stroke check first
missed a genuine strikethrough entirely, then fired on eight labels nothing
touched. This pins the behaviour down with synthetic figures.

Run:  python test_figure_audit.py      (no torch, ~1s)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

import figure_audit as FA

FAILS = []


def check(name, issues, want_substr, should_fire):
    hit = any(want_substr in i for i in issues)
    ok = hit == should_fire
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        expected {want_substr!r} "
              f"{'present' if should_fire else 'absent'}; got {issues}")
        FAILS.append(name)


def base_ax():
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    return fig, ax


print("stroke through text")

# an arrow driven through the middle of a label -> must fire
fig, ax = base_ax()
ax.text(50, 20, "ITA equation", fontsize=12, ha="center", va="center")
ax.add_patch(FancyArrowPatch((20, 20), (55, 20), arrowstyle="-|>",
                             mutation_scale=11, color="k", shrinkA=0, shrinkB=0))
check("arrow through the glyphs fires", FA.audit("t", fig), "STROKE THROUGH TEXT", True)
plt.close(fig)

# a reference line grazing the label's baseline -> must stay quiet
fig, ax = base_ax()
t = ax.text(50, 20, "140", fontsize=12, ha="center", va="bottom")
ax.plot([20, 80], [20, 20], color="k", lw=1)
check("line grazing the baseline stays quiet", FA.audit("t", fig),
      "STROKE THROUGH TEXT", False)
plt.close(fig)

# an arrow stopping short of the label -> must stay quiet
fig, ax = base_ax()
ax.text(50, 20, "ITA equation", fontsize=12, ha="center", va="center")
ax.add_patch(FancyArrowPatch((10, 20), (30, 20), arrowstyle="-|>",
                             mutation_scale=11, color="k", shrinkA=0, shrinkB=0))
check("arrow stopping short stays quiet", FA.audit("t", fig),
      "STROKE THROUGH TEXT", False)
plt.close(fig)

print("container border")

# a label hanging out of a dashed container -> must fire
fig, ax = base_ax()
ax.add_patch(Rectangle((10, 8), 60, 20, fc="none", ec="r", ls="--"))
ax.text(66, 26, "Linear 3->16->32", fontsize=11, ha="center", va="center")
check("label overrunning a container fires", FA.audit("t", fig),
      "CROSSES CONTAINER BORDER", True)
plt.close(fig)

# the same label safely inside -> must stay quiet
fig, ax = base_ax()
ax.add_patch(Rectangle((10, 8), 60, 20, fc="none", ec="r", ls="--"))
ax.text(40, 18, "Linear 3->16->32", fontsize=11, ha="center", va="center")
check("label inside a container stays quiet", FA.audit("t", fig),
      "CROSSES CONTAINER BORDER", False)
plt.close(fig)

# a tick label outside the axes must not count as crossing the axes background
fig, ax = plt.subplots(figsize=(6, 3))
ax.barh([0, 1], [3, 4])
ax.set_yticks([0, 1])
ax.set_yticklabels(["Bal. baseline", "Full"])
check("tick label outside the axes stays quiet", FA.audit("t", fig),
      "CROSSES CONTAINER BORDER", False)
plt.close(fig)

print("chrome is not a stroke")

# gridlines pass behind labels by design, and matplotlib keeps "phantom" ones
# for ticks outside the view -- which land in odd places such as the title band
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot([0, 1], [0, 1])
ax.grid(True)
ax.set_ylim(0, 1.02)
ax.set_title("The specificity deficit holds at every threshold", loc="left")
ax.set_xlabel("Decision threshold t")
check("gridlines do not count as strokes", FA.audit("t", fig),
      "STROKE THROUGH TEXT", False)
plt.close(fig)

print("arrow endpoints")

# an arrow beginning inside a label's glyphs -> must fire
fig, ax = base_ax()
ax.text(50, 20, "180/pi", fontsize=12, ha="center", va="center")
ax.add_patch(FancyArrowPatch((50, 20), (80, 20), arrowstyle="-|>",
                             mutation_scale=11, color="k", shrinkA=0, shrinkB=0))
check("arrow starting inside a label fires", FA.audit("t", fig),
      "ARROW ENDPOINT INSIDE TEXT", True)
plt.close(fig)

# an arrow meeting a box is normal -> must stay quiet
fig, ax = base_ax()
ax.add_patch(Rectangle((60, 12), 25, 16, fc="none", ec="k"))
ax.text(72, 20, "encoding", fontsize=11, ha="center", va="center")
ax.add_patch(FancyArrowPatch((40, 20), (60, 20), arrowstyle="-|>",
                             mutation_scale=11, color="k", shrinkA=0, shrinkB=0))
check("arrow meeting a box stays quiet", FA.audit("t", fig),
      "ARROW ENDPOINT INSIDE TEXT", False)
plt.close(fig)

print()
if FAILS:
    raise SystemExit(f"{len(FAILS)} check(s) behaving wrongly: {', '.join(FAILS)}")
print("all layout checks behave correctly")
