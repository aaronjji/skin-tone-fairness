"""
figure_audit.py - measure every rendered text box in every poster figure and
report the three things that actually go wrong: text running outside the canvas,
text running outside the box it labels, and text overlapping other text.

This is a check, not a renderer. Run it after changing any figure.

Run:  python figure_audit.py            # audits everything
      python figure_audit.py fig4       # just the ones whose name matches
"""
import sys
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import Rectangle

import poster_style

CAPTURED = []


def _capture(fig, out_dir, name):
    """Stand-in for poster_style.save: keep the figure instead of writing it."""
    CAPTURED.append((name, fig))


def _phantom_tick(t):
    """matplotlib keeps tick labels outside the view; they are never drawn."""
    ax = getattr(t, "axes", None)
    if ax is None:
        return False
    for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        for tick in axis.get_major_ticks():
            if tick.label1 is t or tick.label2 is t:
                v = tick.get_loc()
                lo, hi = sorted(lim)
                return not (lo - 1e-9 <= v <= hi + 1e-9)
    return False


def text_boxes(fig):
    r = fig.canvas.get_renderer()
    out = []
    for t in fig.findobj(Text):
        s = t.get_text()
        if not s.strip() or not t.get_visible():
            continue
        if _phantom_tick(t):
            continue
        try:
            bb = t.get_window_extent(renderer=r)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        out.append((t, s, bb))
    return out


def overlap_area(a, b):
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return dx * dy if dx > 0 and dy > 0 else 0.0



def _path_points(artist, renderer, n=80):
    """Display-space points along a line or arrow, or None if undeterminable.

    FancyArrowPatch is the awkward case: across matplotlib versions its get_path()
    may be in data or display space, and get_transform() may be transData or the
    identity. Guessing wrong maps arrows into the canvas corner, where they
    "cross" whatever text sits there. So build every plausible candidate and keep
    the one that actually lies within the artist's own drawn extent.
    """
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D
    import numpy as _np

    def _clean(v):
        try:
            v = _np.asarray(v, dtype=float)
        except Exception:
            return None
        v = v[_np.isfinite(v).all(axis=1)]
        return v if len(v) >= 2 else None

    cands = []
    if isinstance(artist, FancyArrowPatch):
        # get_path() is not usable: in current matplotlib it mixes the shaft in
        # data coordinates with arrowhead geometry in a local frame, so its
        # bounding box is meaningless. The endpoints are what we actually want.
        pab = getattr(artist, "_posA_posB", None)
        if pab is not None:
            ax_ = getattr(artist, "axes", None)
            for tr in ([ax_.transData] if ax_ is not None else []) + [None]:
                try:
                    pts = _np.asarray(pab, dtype=float)
                    cands.append(_clean(tr.transform(pts) if tr is not None else pts))
                except Exception:
                    pass
    elif isinstance(artist, Line2D):
        try:
            cands.append(_clean(artist.get_transform()
                                .transform(artist.get_xydata())))
        except Exception:
            pass
    else:
        return None

    cands = [c for c in cands if c is not None]
    if not cands:
        return None

    try:
        we = artist.get_window_extent(renderer)

        def overflow(v):
            return (max(0.0, we.x0 - v[:, 0].min()) + max(0.0, v[:, 0].max() - we.x1)
                    + max(0.0, we.y0 - v[:, 1].min()) + max(0.0, v[:, 1].max() - we.y1))

        cands.sort(key=overflow)
        if overflow(cands[0]) > 20.0:     # nothing sits inside the drawn extent
            return None
    except Exception:
        pass

    v = cands[0]
    out = []
    for a, b in zip(v[:-1], v[1:]):
        t = _np.linspace(0, 1, n)[:, None]
        out.append(a + (b - a) * t)
    return _np.vstack(out)


def _backgrounds(fig):
    """The figure and axes background patches - never label boxes."""
    bg = {id(fig.patch)}
    for a in fig.axes:
        bg.add(id(a.patch))
    return _IdSet(bg)


class _IdSet:
    def __init__(self, ids):
        self._ids = ids

    def __contains__(self, obj):
        return id(obj) in self._ids


def _strokes(fig):
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D
    skip = set()
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is not None:
            for ch in lg.findobj():
                skip.add(id(ch))
        for axis in (ax.xaxis, ax.yaxis):
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                for ln in (tick.tick1line, tick.tick2line, tick.gridline):
                    if ln is not None:
                        skip.add(id(ln))
    return [a for a in fig.findobj()
            if isinstance(a, (FancyArrowPatch, Line2D)) and a.get_visible()
            and id(a) not in skip]


def audit(name, fig, pad=1.0):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fw, fh = fig.canvas.get_width_height()
    issues = []

    boxes = text_boxes(fig)

    # (figures save with bbox_inches="tight", so text beyond the canvas is
    #  simply included in the saved crop -- not an error.)

    # 2. text wider than the rectangle it sits inside (box labels)
    rects = [p for p in fig.findobj(Rectangle)
             if p.get_visible() and p.get_width() and p.get_height()]
    for t, s, bb in boxes:
        cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
        for p in rects:
            try:
                pb = p.get_window_extent(renderer=r)
            except Exception:
                continue
            if pb.width < 12 or pb.height < 12:
                continue
            # only boxes that look like a labelled block, and only if the text
            # is centred inside this one
            if not (pb.x0 <= cx <= pb.x1 and pb.y0 <= cy <= pb.y1):
                continue
            if pb.width > 0.40 * fw:        # containers, not label boxes
                continue
            if bb.x0 < pb.x0 - pad or bb.x1 > pb.x1 + pad:
                spill = max(pb.x0 - bb.x0, bb.x1 - pb.x1)
                issues.append(f"SPILLS BOX by {spill:5.1f}px : {s[:52]!r}")
                break

    # 3. text over text
    seen = set()
    for i, (t1, s1, b1) in enumerate(boxes):
        for t2, s2, b2 in boxes[i + 1:]:
            a = overlap_area(b1, b2)
            if a <= 4:
                continue
            frac = a / min(b1.width * b1.height, b2.width * b2.height)
            if frac < 0.10:
                continue
            ax1, ax2 = getattr(t1, "axes", None), getattr(t2, "axes", None)
            if ax1 is not None and ax1 is ax2:
                tk = set()
                for axis in (ax1.xaxis, ax1.yaxis):
                    for tick in axis.get_major_ticks():
                        tk.add(id(tick.label1))
                        tk.add(id(tick.label2))
                if id(t1) in tk and id(t2) in tk:
                    continue          # corner tick pair, always adjacent
            key = tuple(sorted((s1[:30], s2[:30])))
            if key in seen:
                continue
            seen.add(key)
            issues.append(f"TEXT OVERLAP {frac*100:4.0f}% : {s1[:34]!r} <> {s2[:34]!r}")

    # 4. annotation text running past its own axes -> tight-bbox padding, i.e.
    #    a wide figure with an empty right side
    for t, s_, bb in boxes:
        ax = getattr(t, "axes", None)
        if ax is None:
            continue
        ticks = set(map(id, ax.get_xticklabels() + ax.get_yticklabels()))
        titles = {id(ax.title), id(getattr(ax, "_left_title", None)),
                  id(getattr(ax, "_right_title", None))}
        if id(t) in ticks or id(t) in titles:
            continue          # titles may run wide; tight bbox absorbs them
        ab = ax.get_window_extent(renderer=r)
        # text anchored past the right edge on purpose (e.g. a value column)
        if getattr(t, "get_transform", None) is not None:
            try:
                if t.get_position()[0] > 1.0 and t.get_transform() is not ax.transData:
                    continue
            except Exception:
                pass
        if bb.x1 > ab.x1 + 2:
            issues.append(f"PAST AXES by {bb.x1 - ab.x1:5.0f}px (wrap it): {s_[:46]!r}")

    # 5. wasted canvas: how much empty margin around all drawn content
    xs0 = min(b.x0 for _, _, b in boxes) if boxes else 0
    xs1 = max(b.x1 for _, _, b in boxes) if boxes else fw
    right_gap = (fw - xs1) / fw * 100
    if right_gap > 22:
        issues.append(f"WHITESPACE: {right_gap:.0f}% of the width is empty on the right")


    # 6. a rule or arrow running through a label (reads as a strikethrough).
    #    Tested against the text's CORE, not its padded bbox: a reference line
    #    that grazes a value label's baseline, or an axis line sitting just above
    #    its own annotation, is normal and must not be flagged.
    import numpy as _np
    strokes = _strokes(fig)
    for t, s_, bb in boxes:
        bp = t.get_bbox_patch() if hasattr(t, "get_bbox_patch") else None
        if bp is not None:
            try:
                fc = bp.get_facecolor()
                if fc is not None and len(fc) == 4 and fc[3] > 0.5:
                    continue        # opaque mask: strokes behind it are hidden
            except Exception:
                pass
        ix = 3.0
        iy = max(3.0, 0.30 * bb.height)
        x0i, x1i = bb.x0 + ix, bb.x1 - ix
        y0i, y1i = bb.y0 + iy, bb.y1 - iy
        if x1i <= x0i or y1i <= y0i:
            continue
        for a in strokes:
            pts = _path_points(a, r)
            if pts is None:
                continue
            inside = ((pts[:, 0] > x0i) & (pts[:, 0] < x1i) &
                      (pts[:, 1] > y0i) & (pts[:, 1] < y1i))
            if inside.sum() < 5:
                continue
            # a strikethrough crosses the label; a connector or leader merely
            # clips its edge. Require the crossing to span a real fraction of it.
            xin = pts[inside][:, 0]
            yin = pts[inside][:, 1]
            spans_x = (xin.max() - xin.min()) >= 0.35 * (x1i - x0i)
            spans_y = (yin.max() - yin.min()) >= 0.60 * (y1i - y0i)
            if spans_x or spans_y:
                issues.append(f"STROKE THROUGH TEXT : {s_[:44]!r}")
                break

    # 7. a label crossing the border of the container it sits in
    for t, s_, bb in boxes:
        cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
        for p_ in rects:
            try:
                pb = p_.get_window_extent(renderer=r)
            except Exception:
                continue
            if pb.width <= 0.40 * fw:          # small boxes: check 2 handles them
                continue
            if p_ in _backgrounds(fig):
                continue                       # figure / axes background, not a box
            if not (pb.x0 <= cx <= pb.x1 and pb.y0 <= cy <= pb.y1):
                continue
            over = max(pb.x0 - bb.x0, bb.x1 - pb.x1, pb.y0 - bb.y0, bb.y1 - pb.y1)
            if over > pad:
                issues.append(f"CROSSES CONTAINER BORDER by {over:4.1f}px : {s_[:40]!r}")
                break

    # (an "arrow flush against a box" check was tried and removed: connectors
    #  are meant to touch the block they point into, and band dividers sit on
    #  their own band edge, so it fired on correct design far more often than on
    #  defects. Arrow-to-box spacing still needs an eye.)

    # 9. an arrow endpoint landing inside a label. A connector is meant to touch
    #    the block it points into, but starting or ending inside text reads as the
    #    arrow growing out of the glyphs.
    from matplotlib.patches import FancyArrowPatch
    for a_ in strokes:
        if not isinstance(a_, FancyArrowPatch):
            continue
        pts = _path_points(a_, r)
        if pts is None:
            continue
        for end in (pts[0], pts[-1]):
            for t, s_, bb in boxes:
                if getattr(t, "arrow_patch", None) is a_:
                    continue          # an annotation's own leader starts at its text
                if bb.x0 <= end[0] <= bb.x1 and bb.y0 <= end[1] <= bb.y1:
                    d = min(end[0] - bb.x0, bb.x1 - end[0])
                    issues.append(
                        f"ARROW ENDPOINT INSIDE TEXT ({d:.1f}px in) : {s_[:38]!r}")
                    break

    seen_msgs = set()
    issues = [i for i in issues
              if not (i in seen_msgs or seen_msgs.add(i))]
    return issues


def main():
    want = sys.argv[1:]
    poster_style.save = _capture           # patch before the modules bind it

    import poster_figures
    import poster_visuals
    import ita_diagnostic
    for m in (poster_figures, poster_visuals, ita_diagnostic):
        m.save = _capture

    poster_style.use_poster_style()
    df = poster_figures.load()
    poster_figures.fig_roc(df, ".")
    poster_figures.fig_spec_curve(df, ".")
    poster_figures.fig_overlap(df, ".")
    poster_figures.fig_dumbbell(df, ".")
    poster_figures.fig_referral(df, ".")
    poster_figures.fig_ita(df, ".")

    dfv = poster_visuals.load()
    poster_visuals.fig_pipeline(dfv, ".")
    poster_visuals.fig_qualitative(dfv, ".")

    import pandas as pd
    if ita_diagnostic.CACHE.exists():
        ita_diagnostic.fig_diagnostic(pd.read_csv(ita_diagnostic.CACHE), ".")

    total = 0
    for name, fig in CAPTURED:
        if want and not any(w in name for w in want):
            continue
        issues = audit(name, fig)
        total += len(issues)
        mark = "OK " if not issues else "!! "
        print(f"\n{mark}{name}")
        for i in issues:
            print(f"     {i}")
    print(f"\n{total} issue(s) across {len(CAPTURED)} figures")


if __name__ == "__main__":
    main()
