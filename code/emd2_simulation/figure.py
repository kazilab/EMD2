"""Figure: the EMD2 chain, simulated.

Same conventions as the EMD1 and EMD4 figures: exact 183 mm double-column
canvas, shared type scale with a 6 pt floor, Type 42 fonts, colour by role,
solid tints rather than alpha, vector + 600 dpi raster export.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .conditions import ARMS, BY_KEY, RATIO_SEGMENT
from .model import SEGMENTS
from .identifiability import load_design

MM = 1 / 25.4
W_DOUBLE = 183 * MM

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans",
                        "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "mathtext.default": "regular",
    "figure.dpi": 160, "savefig.dpi": 600,
    "savefig.bbox": None, "savefig.pad_inches": 0,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.transparent": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
})

INK, INK2, MUTED, RULE = "#12253a", "#3d4a58", "#6b7785", "#c9d1da"
NOSAMPLE = "#f0dcdf"      # cells with no samples in that arm
MICRO, HOST, FLAG = "#1baf7a", "#2a78d6", "#b3323f"
TS = {"fignote": 6.6, "body": 7.2, "label": 7.6, "head": 8.6, "title": 9.4}

ARM_COLOR = {"B": MICRO, "BA": "#e0a030", "Germ-Free": HOST,
             "Monocolonized": "#4fc79b", "2Mem": FLAG, "3Mem": "#7fd4b4"}
SHORT = {"B": "Conv.", "BA": "Abx", "Germ-Free": "GF",
         "Monocolonized": "Mono", "2Mem": "2Mem", "3Mem": "3Mem"}


def tint(hex_color: str, frac: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(c + (255 - c) * (1 - frac)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _style(ax, title, ylabel, tag, xlabel=None):
    ax.set_title(title, fontsize=TS["label"], color=INK, pad=3.5, loc="left")
    ax.set_ylabel(ylabel, fontsize=TS["fignote"], color=INK2, labelpad=2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=TS["fignote"], color=INK2, labelpad=2)
    ax.tick_params(labelsize=TS["fignote"], colors=INK2, pad=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    # Stashed on the axes so make_figure can assert the header block clears
    # the tags. An overlapping "a" is invisible to every numerical check.
    ax._panel_tag = ax.text(-0.24, 1.16, tag, transform=ax.transAxes,
                            fontsize=TS["head"], fontweight="bold", color=INK,
                            va="top", ha="left")


def _band(ens, key, obs, scale=1.0):
    """ABSOLUTE (p5, p50, p95) from the sensitivity band.

    Returns the percentiles themselves rather than offsets. An earlier version
    returned `p50 - p5` and `p95 - p50` as errorbar offsets, which the panels
    then attached to the NOMINAL value -- so whenever nominal differed from the
    median, as it generally does, the drawn endpoints were not p5 and p95. The
    conventional caecum/plasma band is 74.65 / 137.78 / 260.96 against a
    nominal 141.76, which displaced that whisker by about 3.98. The panels now
    draw the interval where it actually is and mark the median on it.
    """
    b = ens["bands"][key][obs]
    return b["p5"] * scale, b["p50"] * scale, b["p95"] * scale


def _draw_band(ax, x, lo, mid, hi, color):
    """p5-p95 interval drawn at its true position, with the median marked.

    Deliberately not an errorbar around the bar height: the bar shows the
    nominal parameterisation and the interval shows the ensemble, and conflating
    them is what produced the displaced whiskers.
    """
    ax.vlines(x, lo, hi, color=color, linewidth=0.55, zorder=5)
    for y in (lo, hi):
        ax.hlines(y, x - 0.09, x + 0.09, color=color, linewidth=0.55, zorder=5)
    ax.plot([x], [mid], marker="_", markersize=4.5, color=color,
            markeredgewidth=0.8, zorder=6)


OBSERVED = Path(__file__).parent / "data" / "observed_roje2024.json"

# Model arm key -> the group spelling each supplementary table uses. The tables
# do not use the deposition's arm labels: S5/S6/S7/S13 say BBN and ABX/BBN,
# S15 says GF/GFA and S39 says 2Mem/3Mem.
OBS_GROUP = {"B": "BBN", "BA": "ABX/BBN"}
OBS_ANIMAL = {"Germ-Free": "GF", "2Mem": "2Mem", "3Mem": "3Mem"}


def _observed():
    """Measured BCPN values from the source paper's supplementary tables.

    Deliberately OPTIONAL. The tables are a manual download, so the figure must
    still build without them rather than making the whole simulation depend on
    a file that is not in the repository. Written by
    ``python -m emd2_simulation.fit.observed``; absent, the panels show the
    model alone exactly as before.
    """
    try:
        return json.loads(OBSERVED.read_text())
    except Exception:
        return None


def make_figure(runs, t, ctx, ident, outdir: Path, ens=None, include_observed=False) -> list[Path]:
    end = ctx["end"]
    null = ctx["null"]
    meas = _observed() if include_observed else None

    fig = plt.figure(figsize=(W_DOUBLE, W_DOUBLE * 0.66))
    # top lowered 0.790 -> 0.770 (V7b note) -> 0.755 (Pass 7 notes on marker
    # meaning and measured above-unity ratios). The header now runs to nine
    # lines and both assertions below caught each step; do not raise this back
    # without rerunning and letting them pass.
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.958, top=0.755, bottom=0.105,
                          wspace=0.44, hspace=0.62)

    # --- a: the luminal gradient -----------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(SEGMENTS))
    for key in ("B", "Germ-Free", "2Mem"):
        vals = [end[key][f"c_BCPN_{s}"] for s in SEGMENTS]
        ax.semilogy(x, np.maximum(vals, 1e-5), color=ARM_COLOR[key], linewidth=1.2,
                    marker="o", markersize=2.4, label=SHORT[key])
    ax.axhline(end["B"]["c_BCPN_plasma"], color=MUTED, linewidth=0.7,
               linestyle=(0, (2, 2)))
    ax.text(0.05, end["B"]["c_BCPN_plasma"] * 1.5, "plasma",
            fontsize=TS["fignote"] - 0.8, color=MUTED)
    if meas is not None:
        # The measured conventional profile, on the same log axis. Two things
        # are meant to be visible at once: the model sits ~3 orders of
        # magnitude high, and its peak is at the COLON where the measured peak
        # is at the caecum. The second is the substantive disagreement.
        pr = meas["profile_uM_3wk"].get("BBN", {})
        ov = [pr.get(s) for s in SEGMENTS]
        if any(v for v in ov if v):
            ax.semilogy(x, [max(v or 0.0, 1e-5) for v in ov], color=INK,
                        linewidth=1.0, linestyle=(0, (1.6, 1.4)), marker="s",
                        markersize=2.6, markerfacecolor="white",
                        markeredgewidth=0.7, zorder=6, label="measured, conv.")
    ax.set_xticks(x)
    ax.set_xticklabels([s[:3] for s in SEGMENTS], fontsize=TS["fignote"] - 0.6)
    _style(ax, "BCPN along the gut", "nmol/mL (log)", "a")
    ax.legend(fontsize=TS["fignote"] - 0.8, frameon=False, loc="lower right",
              handlelength=1.2, borderaxespad=0.2, labelspacing=0.22)

    # --- b: the two-member trap ------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    keys = [a.key for a in ARMS]
    base = end["B"]["AUC_uro"]
    xb = np.arange(len(keys))
    ax.bar(xb - 0.20, [end[k]["AUC_uro"] / base * 100 for k in keys], width=0.38,
           color=[tint(ARM_COLOR[k], 0.55) for k in keys], edgecolor="none",
           label="total index")
    if ens is not None:
        # Sensitivity whiskers on the ratio each arm bears to conventional --
        # the quantity the panel actually claims. The band is taken on the
        # WITHIN-DRAW ratio, not on each arm's marginal AUC: the arms move
        # together under a shared draw, so marginal bands would overstate the
        # spread of a ratio several-fold.
        for j, k in enumerate(keys):
            lo, mid, hi = _band(ens, k, "AUC_rel_conventional", 100.0)
            _draw_band(ax, xb[j] - 0.20, lo, mid, hi, INK2)
    ax.bar(xb + 0.20, [end[k]["micro_fraction"] * 100 for k in keys], width=0.38,
           color=[ARM_COLOR[k] for k in keys], edgecolor="none",
           label="microbial share")
    ax.set_xticks(xb)
    ax.set_xticklabels([SHORT[k] for k in keys], fontsize=TS["fignote"] - 0.8,
                       rotation=30, ha="right")
    ax.axhline(100, color=RULE, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
    # "exposure index", not "AUC": there is no urothelial compartment and the
    # quantity is cumulative urinary excretion times a constant.
    _style(ax, "Exposure index misranks 2Mem", "% (of conv. / of index)", "b")
    ax.legend(fontsize=TS["fignote"] - 1.0, frameon=False, loc="upper left",
              handlelength=1.0, borderaxespad=0.2, labelspacing=0.2)

    # --- c: the discriminating statistic ---------------------------------
    ax = fig.add_subplot(gs[0, 2])
    ratios = [max(end[k]["ratio_cecum"], 1e-3) for k in keys]
    ax.bar(xb, ratios, width=0.62, color=[ARM_COLOR[k] for k in keys],
           edgecolor="none")
    if ens is not None:
        for j, k in enumerate(keys):
            lo, mid, hi = _band(ens, k, "ratio_cecum")
            _draw_band(ax, xb[j], max(lo, 1e-3), max(mid, 1e-3),
                       max(hi, 1e-3), INK2)
    ax.set_yscale("log")
    ax.axhline(1.0, color=FLAG, linewidth=0.8, linestyle=(0, (2, 2)))
    # Sits in the gap between the Abx bar (ends x=1.31) and the jittered GF
    # points (start x=1.85). The old right-edge placement now collides with the
    # measured 3Mem cloud.
    ax.text(1.5, 1.3, "unity", fontsize=TS["fignote"] - 0.8,
            color=FLAG, ha="center")
    if meas is not None:
        # Measured ratios as INDIVIDUAL ANIMALS, not medians. The overlap
        # between the converting and non-converting consortia is the finding --
        # a median-only overlay would hide exactly what matters. Conventional
        # and antibiotic summaries use batch-matched cohort medians; this
        # importer has not established cross-sheet animal-ID linkage.
        jit = np.random.default_rng(0)
        for j, k in enumerate(keys):
            if k in OBS_ANIMAL:
                pts = meas["ratio_per_animal"].get(OBS_ANIMAL[k], [])
            elif k in OBS_GROUP:
                pts = [r["ratio"] for r in
                       meas["ratio_batch_matched"].get(OBS_GROUP[k], {}).values()]
            else:
                pts = []                      # mono-colonised: no comparator
            if not pts:
                continue
            # Filled circles are SINGLE ANIMALS (same-animal caecum/plasma
            # pairs). Open squares are BATCH-MATCHED COHORT MEDIANS -- the
            # conventional and antibiotic arms use cohort summaries,
            # from separately reported caecum and plasma tables.
            # Drawing both with one marker under a legend reading "single
            # animals", as earlier releases did, mislabels four of the points.
            per_animal = k in OBS_ANIMAL
            dx = (jit.random(len(pts)) - 0.5) * 0.30
            ax.scatter(np.full(len(pts), j) + dx, [max(p, 1e-3) for p in pts],
                       s=4.6 if per_animal else 9.0,
                       marker="o" if per_animal else "s",
                       color=INK if per_animal else "none",
                       edgecolor="white" if per_animal else INK,
                       linewidth=0.3 if per_animal else 0.7,
                       zorder=7)
    ax.set_xticks(xb)
    ax.set_xticklabels([SHORT[k] for k in keys], fontsize=TS["fignote"] - 0.8,
                       rotation=30, ha="right")
    # NOT "attributes origin": V7b shows a host-only structure reaching 1.37,
    # and the measured consortia do not separate (p = 0.13).
    _style(ax, "Caecum/plasma ratio by arm", "ratio (log)", "c")

    # --- d: the null ------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    cats = ["urinary\nBCPN", "caecal\nBCPN", "caecum/plasma\nratio"]
    mech = [null["urine_target"], null["cecum_mech"], end["B"]["ratio_cecum"]]
    nullv = [null["urine_null"], null["cecum_null"], max(null["ratio"], 1e-3)]
    xd = np.arange(3)
    ax.bar(xd - 0.19, mech, width=0.36, color=MICRO, edgecolor="none",
           label="microbial model")
    ax.bar(xd + 0.19, nullv, width=0.36, color=FLAG, edgecolor="none",
           label="host-only null")
    if meas is not None:
        # Measured counterparts of the same three quantities. Urine is reported
        # in nM and converted to nmol/mL to match the model's units; the ratio
        # is the median of the two batch-matched cohort values.
        rr = [r["ratio"] for r in meas["ratio_batch_matched"]["BBN"].values()]
        obs_d = [meas["urine_3wk"]["conventional"]["nM"] / 1000.0,
                 meas["profile_uM_3wk"]["BBN"]["cecum"],
                 float(np.median(rr))]
        for i, m in enumerate(obs_d):
            ax.hlines(m, i - 0.34, i + 0.34, color=INK, linewidth=1.3, zorder=7)
        ax.plot([], [], color=INK, linewidth=1.3, label="measured")
    ax.set_yscale("log")
    ax.set_xticks(xd)
    ax.set_xticklabels(cats, fontsize=TS["fignote"] - 0.8)
    # NOT an R^2: three free parameters recovering one scalar exactly.
    # "simulated urine": the target is this model's own output, not a measured
    # value, so the caecal gap is a model-model disagreement.
    _style(ax, "Null recovers simulated urine", "value (log)", "d")
    ax.legend(fontsize=TS["fignote"] - 0.8, frameon=False, loc="upper right",
              handlelength=1.0, borderaxespad=0.2, labelspacing=0.2)

    # --- e: identifiability ----------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    if ident is not None:
        sets = [("urine\nonly", "urine_only"),
                ("urine +\nlumen", "urine_plus_lumen"),
                ("lumen,\n3 arms", "lumen_all_arms"),
                ("kidney,\n6 arms", "kidney_all_arms"),
                ("lumen +\nkidney", "lumen_plus_kidney")]
        widths = [ident[k]["ridge_width"] * 100 for _, k in sets]
        sat = [ident[k]["saturated"] for _, k in sets]
        cols = [FLAG if t_ else ("#e0a030" if w > 90 else MICRO)
                for w, t_ in zip(widths, sat)]
        xe = np.arange(len(sets))
        ax.bar(xe, widths, width=0.62, color=cols, edgecolor="none")
        ax.set_xticks(xe)
        ax.set_xticklabels([s for s, _ in sets], fontsize=TS["fignote"] - 1.4)
        for i, (w, t_) in enumerate(zip(widths, sat)):
            # A saturated ridge runs off the tested grid. Printing its width as
            # a percentage would report a property of the grid, not the design.
            ax.text(i, w + max(widths) * 0.035,
                    "unbounded" if t_ else f"{w:.0f}%",
                    fontsize=TS["fignote"] - 1.4,
                    color=FLAG if t_ else INK2, ha="center",
                    fontweight="bold" if t_ else "normal")
        ax.set_ylim(0, max(widths) * 1.28)
    else:
        # A blank panel is a defect, not a neutral outcome: it looks like a
        # rendering failure rather than a deliberate omission. Say why it is
        # empty on the panel itself.
        ax.text(0.5, 0.5, "not computed\n(--no-identifiability)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=TS["fignote"] - 0.4, color=MUTED, linespacing=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
    _style(ax, r"Ridge on $V_{max,ox}$ (worst residual)", "% of nominal", "e")

    # --- f: the deposited design -----------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    counts = load_design()
    parts = ["cecum", "colon", "rectum", "ileum", "Plasma", "bladder",
             "bile", "urine", "feces"]
    grid = np.zeros((len(parts), len(keys)))
    for j, k in enumerate(keys):
        for i, pt in enumerate(parts):
            grid[i, j] = sum(counts.get(k, {}).get(pt, {}).values())
    # Zero is the informative value in this panel -- the disjointness argument
    # lives in the empty cells -- so unsampled combinations are masked to a
    # distinct colour rather than being left to read as "merely dark".
    masked = np.ma.masked_where(grid <= 0, np.log1p(grid))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad(NOSAMPLE)
    im = ax.imshow(masked, cmap=cmap, aspect="auto",
                   vmin=0, vmax=np.log1p(grid.max()))
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([SHORT[k] for k in keys], fontsize=TS["fignote"] - 1.0,
                       rotation=30, ha="right")
    ax.set_yticks(np.arange(len(parts)))
    ax.set_yticklabels([p.lower() for p in parts], fontsize=TS["fignote"] - 1.0)
    cb = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.035)
    ticks = [t for t in (1, 5, 20, 50, 130) if t <= grid.max()]
    cb.set_ticks(np.log1p(ticks))
    cb.set_ticklabels([str(t) for t in ticks])
    cb.ax.tick_params(labelsize=TS["fignote"] - 1.4, colors=INK2, length=1.6,
                      width=0.5, pad=1.2)
    cb.ax.set_title("records", fontsize=TS["fignote"] - 1.0, color=INK2,
                    pad=3.0)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor(RULE)
    ax.legend(handles=[Patch(facecolor=NOSAMPLE, edgecolor=RULE, linewidth=0.5,
                             label="not sampled")],
              fontsize=TS["fignote"] - 1.2, frameon=False,
              loc="upper left", bbox_to_anchor=(0.0, -0.16),
              handlelength=1.0, handleheight=0.9, borderaxespad=0.0)
    # Acquisition counts are retained as deposited; no biological n is inferred.
    _style(ax, "MTBLS3581 design", "", "f")
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)

    title = fig.text(0.075, 0.985,
                     "EMD2: illustrative microbiome-mediated toxicokinetics",
                     fontsize=TS["title"], fontweight="bold", color=INK, va="top")
    lines = [
        (0.956, MUTED,
         "KCC1-related metabolism; KCC6 home and KCC7/8/10 links require functional evidence. KCC2 requires genotoxicity evidence."),
        (0.937, MUTED,
         "BBN → BCPN; assigned kinetics and dose (598 nmol/h). Neither concentrations nor ratios are calibrated animal predictions."),
        (0.918, MUTED,
         "The exposure index is proportional to urinary excretion. The terminal ratio does not uniquely identify microbial origin."),
    ]
    if ens is not None:
        lines.append((0.899, MUTED,
                      f"b, c: bars = nominal simulations; adjacent lines = p5–p95 sensitivity ranges over {ens['n_ok']} draws, with median ticks; not CIs."))
    if meas is not None:
        lines.extend([
            (0.880, FLAG, "Observed overlays: Roje 2024 supplementary tables. c: circles = paired animals; squares = ratios of batch-matched cohort medians."),
            (0.861, FLAG, "c: source-table ratios are provisional; matrix-specific normalisation is unresolved. Unity is not a validated physical cutoff."),
            (0.842, FLAG, "Consortium separation was not established (p = 0.13; n = 8 per arm). d: host-only null fitted to simulated urine."),
            (0.823, MUTED, "e: conditional synthetic recoverability. f: LC-MS acquisitions, not animals; source anatomy labels include flagged inconsistencies."),
        ])
    else:
        lines.append((0.880, FLAG, "No observed overlays loaded. e: conditional synthetic recoverability; f: acquisition counts, not independent biological n."))
    handles = [title] + [fig.text(0.075, y, txt, fontsize=TS["fignote"],
                                  color=col, va="top")
                         for y, col, txt in lines]

    # Measure what was actually rendered. matplotlib will happily draw text
    # past the canvas edge; nothing else in this build would catch it.
    fig.canvas.draw()
    right_limit = 0.995 * fig.get_figwidth() * fig.dpi
    over = [h.get_text()[:40] for h in handles
            if h.get_window_extent(fig.canvas.get_renderer()).x1 > right_limit]
    if over:
        raise RuntimeError(
            "figure header text overflows the 183 mm canvas; re-wrap: "
            + "; ".join(over))

    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    header_bottom = min(inv.transform(h.get_window_extent(rend).p0)[1]
                        for h in handles)
    tags = [a._panel_tag for a in fig.axes if hasattr(a, "_panel_tag")]
    tag_top = max(inv.transform(t.get_window_extent(rend).p1)[1] for t in tags)
    if tag_top > header_bottom:
        raise RuntimeError(
            f"header block overlaps the panel tags (tag top {tag_top:.3f} > "
            f"header bottom {header_bottom:.3f}); lower the gridspec top")

    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "svg", "png", "eps", "tif"):
        pth = outdir / f"figureS_emd2_mechanistic.{ext}"
        fig.savefig(pth, format=ext)
        paths.append(pth)
    plt.close(fig)
    print(f"  wrote {paths[0].with_suffix('.{pdf,svg,png,eps,tif}')}")
    return paths
