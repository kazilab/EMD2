"""Figure 4. Model comparison for the anatomical source of BCPN."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..simulation.conditions import ARMS, BY_KEY, RATIO_SEGMENT
from ..simulation.model import SEGMENTS
from ..analysis.identifiability import load_design

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


def _style(ax, ylabel, tag, xlabel=None):
    ax.set_ylabel(ylabel, fontsize=TS["fignote"], color=INK2, labelpad=2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=TS["fignote"], color=INK2, labelpad=2)
    ax.tick_params(labelsize=TS["fignote"], colors=INK2, pad=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax._panel_tag = ax.text(-0.24, 1.16, tag, transform=ax.transAxes,
                            fontsize=TS["head"], fontweight="bold", color=INK,
                            va="top", ha="left")


def _band(ens, key, obs, scale=1.0):
    """Return the 5th, 50th and 95th sensitivity percentiles."""
    b = ens["bands"][key][obs]
    return b["p5"] * scale, b["p50"] * scale, b["p95"] * scale


def _draw_band(ax, x, lo, mid, hi, color):
    """Draw the sensitivity interval and its median."""
    ax.vlines(x, lo, hi, color=color, linewidth=0.55, zorder=5)
    for y in (lo, hi):
        ax.hlines(y, x - 0.09, x + 0.09, color=color, linewidth=0.55, zorder=5)
    ax.plot([x], [mid], marker="_", markersize=4.5, color=color,
            markeredgewidth=0.8, zorder=6)


# Group codes in the source tables.
OBS_GROUP = {"B": "BBN", "BA": "ABX/BBN"}
OBS_ANIMAL = {"Germ-Free": "GF", "2Mem": "2Mem", "3Mem": "3Mem"}


def _observed():
    """Load descriptive comparisons from the source measurements."""
    from ..processing.store import load_observed
    return load_observed()


def create_figure(runs, t, ctx, ident=None, ens=None, include_observed=False, *,
                  show_condition_legend=False):
    """Plot the simulated end points and optional descriptive measurements."""
    end = ctx["end"]
    null = ctx["null"]
    meas = _observed() if include_observed else None

    fig = plt.figure(figsize=(W_DOUBLE, 4.20))
    gs = fig.add_gridspec(
        2, 3, left=0.075, right=0.958,
        top=0.92, bottom=0.14,
        wspace=0.44, hspace=0.62)

    # a. Luminal concentrations
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(SEGMENTS))
    for key in ("B", "Germ-Free", "2Mem"):
        vals = [end[key][f"c_BCPN_{s}"] for s in SEGMENTS]
        ax.semilogy(x, np.maximum(vals, 1e-5), color=ARM_COLOR[key], linewidth=1.2,
                    marker="o", markersize=2.4, label=SHORT[key])
    ax.axhline(end["B"]["c_BCPN_plasma"], color=MUTED, linewidth=0.7,
               linestyle=(0, (2, 2)))
    if meas is not None:
        pr = meas["profile_uM_3wk"].get("BBN", {})
        ov = [pr.get(s) for s in SEGMENTS]
        if any(v for v in ov if v):
            ax.semilogy(x, [max(v or 0.0, 1e-5) for v in ov], color=INK,
                        linewidth=1.0, linestyle=(0, (1.6, 1.4)), marker="s",
                        markersize=2.6, markerfacecolor="white",
                        markeredgewidth=0.7, zorder=6, label="measured, conv.")
    ax.set_xticks(x)
    ax.set_xticklabels([s[:3] for s in SEGMENTS], fontsize=TS["fignote"] - 0.6)
    _style(ax, "BCPN (nmol/mL; log)", "a")
    if show_condition_legend:
        for key, label in (("B", "Conventional"), ("Germ-Free", "Germ-free"),
                           ("2Mem", "Two-member")):
            above = key == "2Mem"
            ax.annotate(label, xy=(x[-1], end[key][f"c_BCPN_{SEGMENTS[-1]}"]),
                        xytext=(-3, 3 if above else -4), textcoords="offset points",
                        ha="right", va="bottom" if above else "top",
                        fontsize=6.2, color=ARM_COLOR[key], clip_on=True)
        ax.annotate("Conv. plasma", xy=(0.025, end["B"]["c_BCPN_plasma"]),
                    xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", fontsize=6.2, color=MUTED,
                    va="bottom", clip_on=True)
        if meas is not None:
            measured_lines = [line for line in ax.lines if line.get_label() == "measured, conv."]
            if measured_lines:
                ax.legend(handles=measured_lines, fontsize=6.2, frameon=False,
                          loc="lower right", handlelength=1.2, borderaxespad=0.2)

    # b. Exposure index and microbial fraction
    ax = fig.add_subplot(gs[0, 1])
    keys = [a.key for a in ARMS]
    base = end["B"]["AUC_uro"]
    xb = np.arange(len(keys))
    ax.bar(xb - 0.20, [end[k]["AUC_uro"] / base * 100 for k in keys], width=0.38,
           color=[tint(ARM_COLOR[k], 0.55) for k in keys], edgecolor="none",
           label="total index")
    if ens is not None:
        # Compute percentiles of within-draw ratios to preserve the shared parameter draw.
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
    _style(ax, "% (of conv. / of index)", "b")
    if show_condition_legend:
        handles = [Patch(facecolor="#b8c2cc", label="Total index"),
                   Patch(facecolor=INK2, label="Microbial share")]
        ax.legend(handles=handles, fontsize=6.2, frameon=False, loc="upper left",
                  handlelength=1.0, borderaxespad=0.2, labelspacing=0.2)
        ax.annotate("100% Conv.", xy=(0.025, 100),
                    xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", fontsize=6.2, color=MUTED,
                    va="bottom", clip_on=True)

    # c. Caecum-to-plasma ratio
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
    if meas is not None:
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
            # Distinguish paired animals from ratios of batch-matched cohort medians.
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
    _style(ax, "Caecum/plasma\nratio (log)", "c")
    if show_condition_legend:
        ax.annotate("Unity", xy=(2, 1), xytext=(0, 3),
                    textcoords="offset points", fontsize=6.2, color=FLAG,
                    ha="center", va="bottom", clip_on=True)

    # d. Host-only comparison
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
        # Convert urinary nM to nmol/mL. Summarize batch-matched ratios by their median.
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
    _style(ax, "value (log)", "d")
    if show_condition_legend:
        ax.legend(fontsize=TS["fignote"] - 0.8, frameon=False, loc="upper right",
                  handlelength=1.0, borderaxespad=0.2, labelspacing=0.2)

    # e. Synthetic profile widths
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
            # A profile reaching the grid boundary has no upper width estimate.
            ax.text(i, w + max(widths) * 0.035,
                    "unbounded" if t_ else f"{w:.0f}%",
                    fontsize=TS["fignote"] - 1.4,
                    color=FLAG if t_ else INK2, ha="center",
                    fontweight="bold" if t_ else "normal")
        ax.set_ylim(0, max(widths) * 1.28)
    else:
        ax.text(0.5, 0.5, "Not computed",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=TS["fignote"] - 0.4, color=MUTED, linespacing=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
    _style(ax, "Profile width\n(% of nominal)", "e")

    # f. Acquisition coverage
    ax = fig.add_subplot(gs[1, 2])
    counts = load_design()
    parts = ["cecum", "colon", "rectum", "ileum", "Plasma", "bladder",
             "bile", "urine", "feces"]
    grid = np.zeros((len(parts), len(keys)))
    for j, k in enumerate(keys):
        for i, pt in enumerate(parts):
            grid[i, j] = sum(counts.get(k, {}).get(pt, {}).values())
    # Mask unsampled cells before applying the logarithmic colour scale.
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
    if show_condition_legend:
        ax.legend(handles=[Patch(facecolor=NOSAMPLE, edgecolor=RULE, linewidth=0.5,
                                 label="not sampled")],
                  fontsize=TS["fignote"] - 1.2, frameon=False,
                  loc="upper left", bbox_to_anchor=(0.0, -0.16),
                  handlelength=1.0, handleheight=0.9, borderaxespad=0.0)
    _style(ax, "", "f")
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    return fig


def write_legend(outdir: Path, *, ens=None, ident=None, include_observed=False) -> Path:
    """Write the caption for Figure 4."""
    text = """# Figure 4. Model comparison for the anatomical source of BCPN

Predictions use assigned compartmental kinetics. Conv. denotes conventional animals,
Abx antibiotic-depleted animals, GF germ-free animals and Mono monocolonised animals.
2Mem and 3Mem denote two- and three-member communities.

**a** Simulated BCPN concentrations along the gut. The grey dashed line marks the
conventional plasma concentration.

**b** Pale bars show the cumulative exposure index as a percentage of the conventional
value. Dark bars show the percentage of each arm’s index attributed to microbial
formation. The index is proportional to cumulative urinary excretion. Source
fractions are model estimates.

**c** Caecum-to-plasma concentration ratio by arm. The dashed line marks equal
concentrations. Host transport can also produce ratios above one.

**d** The microbial model (green) and a host-only model (red) give the same urinary
endpoint under different assumptions about BCPN formation. The host-only model was
fitted to simulated urine. Urinary and caecal concentrations are in nmol/mL.
"""
    if ident is None:
        text += "\n**e** Profile widths were not computed for this run.\n"
    else:
        text += """
**e** Profile width is the range of microbial oxidation capacities consistent with
the synthetic measurements, expressed as a percentage of nominal capacity. At each
capacity, the model refits deconjugation capacity and holds the other kinetics fixed.
The reference analysis admits values for which every prediction falls within 15% of
its synthetic target. Red bars reach a grid boundary. Gold marks bounded widths
above 90% and green marks bounded widths up to 90%.
"""
    text += """
**f** LC–MS acquisition records by compartment and arm. Pink cells indicate
unsampled combinations. Each count is an acquisition. [MetaboLights MTBLS3581](https://www.ebi.ac.uk/metabolights/MTBLS3581)
"""
    if ens is not None:
        text += (f"\nWhiskers in b and c span the 5th–95th percentiles across {ens['n_ok']} "
                 "accepted parameter draws. Horizontal ticks mark the medians.\n")
    text += ("\nThe predicted concentrations and the position of the peak along the gut "
             "differ from the published measurements. "
             "[Roje et al. (2024)](https://doi.org/10.1038/s41586-024-07754-w)\n")
    if include_observed:
        text += """
Open squares and dashed lines in a show the published conventional profile. In c,
filled circles show paired animals and open squares show ratios of batch-matched
cohort medians. Conflicting germ-free caecal measurements were excluded. Black
lines in d mark the corresponding measurements. Cross-matrix comparisons depend
on unresolved differences in assay normalisation.
"""
    path = Path(outdir) / "emd2_mechanistic_legend.md"
    path.write_text(text, encoding="utf-8")
    return path


def make_figure(runs, t, ctx, ident, outdir: Path, ens=None, include_observed=False) -> list[Path]:
    """Export Figure 4 and its caption."""
    fig=create_figure(runs,t,ctx,ident,ens,include_observed,show_condition_legend=True)
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    paths=[]
    for ext in ("pdf","svg","png","eps","tif"):
        path=outdir/f"emd2_mechanistic.{ext}"
        options={"pil_kwargs":{"compression":"tiff_lzw"}} if ext=="tif" else {}
        fig.savefig(path,**options);paths.append(path)
    plt.close(fig)
    paths.append(write_legend(outdir,ens=ens,ident=ident,include_observed=include_observed))
    return paths
