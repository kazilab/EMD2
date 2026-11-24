"""Figures 3 and S5. Published BCPN measurements and controls."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.lines import Line2D

from ..analysis.empirical import TISSUE_LABEL

INK, MUTED, GREEN, BLUE, GOLD = "#172c42", "#647385", "#139573", "#356cb5", "#c28c29"
STYLE = {"font.family": "DejaVu Sans", "font.size": 8,
         "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7,
         "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
         "axes.spines.right": False, "axes.linewidth": 0.6,
         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}


def _style(ax, tag, title):
    ax.set_title(title, loc="left", color=INK, pad=9)
    ax.text(-0.21, 1.075, tag, transform=ax.transAxes, fontsize=11, weight="bold", color=INK)
    ax.tick_params(width=0.6, length=3)
    ax.spines["left"].set_color("#a9b3bd")
    ax.spines["bottom"].set_color("#a9b3bd")


def _forest(ax, rows, labels, xlabel, *, show_legend=False):
    for y, row in enumerate(rows):
        value, ci = row["mean_ratio"], row["ratio_ci95"]
        if value is None or value <= 0:
            continue
        colour = GREEN if row["p_holm"] < 0.05 else BLUE
        if ci is not None and ci[0] > 0:
            ax.plot(ci, [y, y], color=colour, lw=1.4)
            ax.scatter(value, y, c=colour, s=21, zorder=3)
        else:
            ax.scatter(value, y, marker="D", facecolors="white", edgecolors=colour, s=24, zorder=3)
    ax.axvline(1, color="#8a96a3", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#e9edf1", lw=0.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    if show_legend:
        handles = [Line2D([], [], color=GREEN, marker="o", lw=1.2, ms=3, label="Holm P < 0.05"),
                   Line2D([], [], color=BLUE, marker="o", lw=1.2, ms=3, label="Holm P ≥ 0.05")]
        ax.legend(handles=handles, frameon=False, ncols=2, fontsize=6,
                  loc="lower left", bbox_to_anchor=(0, -0.48),
                  handlelength=1.2, columnspacing=0.8, borderaxespad=0)


def _acute(ax, result, tissue, *, show_legend=False):
    for group, colour, marker in (("BBN", GREEN, "o"), ("ABX/BBN", GOLD, "s")):
        rows = sorted((r for r in result["acute"]["descriptive"] if r["group"] == group and r["tissue"] == tissue), key=lambda r: r["time_h"])
        for r in rows:
            offsets = np.linspace(-0.12, 0.12, len(r["values_uM"]))
            # EPS requires opaque marker colours.
            point_colour = tuple(0.55 * value + 0.45 for value in matplotlib.colors.to_rgb(colour))
            ax.scatter(r["time_h"] + offsets, r["values_uM"], s=11, c=[point_colour],
                       marker=marker, edgecolors="none")
        ax.plot([r["time_h"] for r in rows], [r["mean_uM"] for r in rows],
                marker=marker, ms=3.5, color=colour, label=group, lw=1.2)
    ax.set_xticks([1, 3, 6, 9])
    ax.set_xlabel("Hours after oral BBN bolus")
    ax.set_ylabel("BCPN (source-table µM)")
    ax.set_ylim(bottom=-5)
    if show_legend:
        ax.legend(frameon=False)


def _save(fig, outdir, stem):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    # Check all title, axis-label and figure-note extents against the canvas.
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label]
        texts += list(ax.texts)
        # Matplotlib also creates tick labels outside the displayed limits.
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())),
                               (ax.yaxis, sorted(ax.get_ylim()))):
            index = 0 if axis is ax.xaxis else 1
            for label in axis.get_ticklabels():
                position = label.get_position()[index]
                if lo - 1e-9 <= position <= hi + 1e-9:
                    texts.append(label)
    legends = list(fig.legends) + [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    for legend in legends:
        box = legend.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
            raise RuntimeError("Figure key outside canvas")
    for text in texts:
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
            raise RuntimeError(f"Figure text outside canvas: {text.get_text()}")
    paths = []
    for ext in ("png", "pdf", "svg", "tif", "eps"):
        path = outdir / f"{stem}.{ext}"
        kwargs = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
        with plt.rc_context(STYLE):
            fig.savefig(path, dpi=450, facecolor="white", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def create_figure(result, *, show_condition_legend=True):
    """Plot donor and animal measurements for Figure 3."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(183/25.4, 5.10), dpi=160)
        fig.subplots_adjust(left=0.085, right=0.985, top=0.94, bottom=0.19,
                            hspace=0.86, wspace=0.68)
        ax = axes[0, 0]
        donor_ids = sorted({r["donor"] for r in result["human_donors"]})
        lookup = {(r["donor"], r["oxygen"]): r["mean_nM"] for r in result["human_donors"]}
        for donor in donor_ids:
            y = [lookup[donor, o] for o in ("0%O2", "10%O2", "21%O2")]
            ax.plot(range(3), y, color="#aab8c6", lw=0.7, marker="o", ms=3,
                    label="Individual donor" if donor == donor_ids[0] else None)
        mean = [np.mean([lookup[k, o] for k in donor_ids]) for o in ("0%O2", "10%O2", "21%O2")]
        ax.plot(range(3), mean, color=GREEN, lw=2, marker="D", ms=4, label="Donor mean")
        ax.set_xticks(range(3), ["0%", "10%", "21%"])
        ax.set_xlabel("Culture oxygen at 24 h")
        ax.set_ylabel("Culture BCPN (nM)")
        if show_condition_legend:
            ax.set_ylim(top=max(max(lookup.values()) * 1.35, 1.0))
            ax.legend(frameon=False, fontsize=6, loc="upper left",
                      handlelength=1.2, borderaxespad=0.2)
        _style(ax, "a", "")

        ax = axes[0, 1]
        row = next(r for r in result["consortium"] if r["Tissue"] == "CEC")
        for x, key, colour, label in ((0, "observations_b", BLUE, "2-member"), (1, "observations_a", GREEN, "3-member")):
            vals = [r["value"] for r in row[key]]
            ax.scatter(x + np.linspace(-0.13, 0.13, len(vals)), vals, s=24, c=colour, edgecolors="white", linewidths=0.4)
            ax.hlines(np.mean(vals), x-0.20, x+0.20, color=colour, lw=2)
        ax.set_xticks([0, 1], ["2-member", "3-member"])
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylabel("Caecal BCPN (µM)")
        ax.set_ylim(0, 6.1)
        _style(ax, "b", "")

        ax = axes[0, 2]
        order = ("SI1", "SI2", "SI3", "CEC", "COL", "REC", "PL", "LVR", "KID")
        rows = [next(r for r in result["consortium"] if r["Tissue"] == k) for k in order]
        labels = ["SI1", "SI2", "SI3", "Caecum", "Colon", "Rectum", "Plasma", "Liver", "Kidney"]
        _forest(ax, rows, labels, "BCPN mean ratio\n3-member / 2-member")
        ax.set_xlim(0.08, 12)
        ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
        _style(ax, "c", "")

        ax = axes[1, 0]
        _acute(ax, result, "Cecum", show_legend=show_condition_legend)
        ax.set_xlabel("Hours after oral\nBBN bolus")
        ax.set_ylabel("Caecal BCPN\n(source-table µM)")
        if show_condition_legend:
            ax.legend(frameon=False, fontsize=6, loc="upper right", handlelength=1.2)
        _style(ax, "d", "")
        ax = axes[1, 1]
        _acute(ax, result, "PL", show_legend=show_condition_legend)
        ax.set_xlabel("Hours after oral\nBBN bolus")
        ax.set_ylabel("Plasma BCPN\n(source-table µM)")
        if show_condition_legend:
            ax.legend(frameon=False, fontsize=6, loc="upper right", handlelength=1.2)
        _style(ax, "e", "")

        ax = axes[1, 2]
        order = [("3wk", "3wk_1213"), ("3wk", "Neg_12wk_0629"), ("6wk", "Neg_12wk_0629"),
                 ("7wk", "Neg_7wk_0211"), ("12wk", "Neg_12wk_0629")]
        rows = [next(r for r in result["urine"] if (r["Length"], r["Batch"]) == key) for key in order]
        _forest(ax, rows, ["3 wk · 1213", "3 wk · 0629", "6 wk · 0629", "7 wk · 0211", "12 wk · 0629"], "24-h urinary amount\nBBN / ABX–BBN")
        ax.set_xlim(0.2, 50)
        ax.set_xticks([0.2, 1, 5, 25], ["0.2", "1", "5", "25"])
        _style(ax, "f", "")
        if show_condition_legend:
            handles = [Line2D([], [], color=GREEN, marker="o", lw=1.2, ms=3, label="Holm P < 0.05"),
                       Line2D([], [], color=BLUE, marker="o", lw=1.2, ms=3, label="Holm P ≥ 0.05")]
            for ax in (axes[0, 2], axes[1, 2]):
                ax.legend(handles=handles, frameon=False, ncols=1, fontsize=6,
                          loc="upper left", bbox_to_anchor=(0, -0.40),
                          handlelength=1.2, borderaxespad=0)
        return fig


def create_controls_figure(result, *, show_condition_legend=True):
    """Plot supporting measurements and controls for Figure S5."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(183/25.4, 5.80 if show_condition_legend else 5.40), dpi=160)
        fig.subplots_adjust(left=0.13, right=0.96, top=0.94,
                            bottom=0.18 if show_condition_legend else 0.12,
                            hspace=0.82 if show_condition_legend else 0.62, wspace=0.46)
        ax = axes[0, 0]
        signals = [r for r in result["isolate_signals"] if r["time_h"] == 12 and r["strain"] == "Escherichia coli ED1a"]
        for x, oxygen in enumerate(("0%O2", "10%O2", "21%O2")):
            row = next((r for r in signals if r["oxygen"] == oxygen), None)
            if row:
                ax.scatter(x + np.linspace(-0.1, 0.1, len(row["signals"])), np.asarray(row["signals"])/1e3, s=20, c=GREEN)
                ax.hlines(row["mean_signal"]/1e3, x-0.17, x+0.17, color=GREEN, lw=1.5)
        ax.set_xticks(range(3), ["0%", "10%", "21%"])
        ax.set_xlabel("Culture oxygen · 12 h")
        ax.set_ylabel("BCPN signal (10³ a.u.)")
        _style(ax, "a", "")

        ax = axes[0, 1]
        _acute(ax, result, "Liver", show_legend=show_condition_legend)
        ax.set_ylabel("Liver BCPN\n(source-table µM)")
        _style(ax, "b", "")

        ax = axes[1, 0]
        rows = result["bladder"]
        _forest(ax, rows, [r["Length"].replace("wk", " wk") + " · " + r["Batch"].split("_")[-1] for r in rows], "Mean bladder BCPN: BBN / ABX–BBN", show_legend=show_condition_legend)
        _style(ax, "c", "")

        ax = axes[1, 1]
        rows = result["germ_free_controls"]
        _forest(ax, rows, [TISSUE_LABEL[r["Tissue"]] for r in rows], "Mean BCPN: GF + ABX / GF", show_legend=show_condition_legend)
        ax.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
        _style(ax, "d", "")
        return fig


def write_legends(result, outdir: Path) -> list[Path]:
    """Write the captions for Figures 3 and S5."""
    main = """# Figure 3. Reanalysis of published BCPN measurements following BBN exposure

**a** Each grey line connects the mean BCPN concentrations for one donor across
culture conditions at 24 h. Green diamonds show the equal-weight mean across ten
donors. Technical wells were averaged within donor before the paired analysis.

**b** Caecal BCPN concentrations measured before deglucuronidation in the
two-member (blue) and three-member (green) communities. Each dot represents one animal (eight per group). Bars show group means.

**c** Ratio of mean BCPN concentrations in three-member versus two-member
communities within each compartment.

**d, e** Acute caecal (d) and plasma (e) measurements. Small points show observations
and connected points show means. Green circles denote BBN alone and gold squares
denote antibiotics plus BBN (ABX/BBN). Discrepancies in treatment labels, counts
and reported statistics limit these panels to descriptive comparisons.

**f** Ratio of mean 24-hour urinary BCPN amounts for BBN versus ABX/BBN within
each collection week and analytical batch.

In c and f, bars show pointwise 95% bootstrap confidence intervals from resampling
animals. Green marks Holm-adjusted P < 0.05 within each family and blue marks
P ≥ 0.05. An open diamond indicates that a zero bootstrap denominator prevented
estimation of a ratio interval.

Source measurements from [Roje et al. (2024)](https://doi.org/10.1038/s41586-024-07754-w).
"""
    controls = """# Figure S5. Supporting measurements and controls for microbial conversion of BBN

**a** BCPN signal from the published *Escherichia coli* ED1a measurements, in thousands of arbitrary units. Points denote technical replicates
and horizontal bars show their means.

**b** Acute liver BCPN measurements. Small points show observations and connected
points show means. Green circles denote BBN alone and gold squares denote antibiotics
plus BBN (ABX/BBN). Treatment-label and sample-count discrepancies limit interpretation.

**c** Ratio of mean bladder BCPN concentrations for BBN versus ABX/BBN within each
collection week and analytical batch.

**d** Ratio of mean BCPN concentrations in germ-free animals given antibiotics
versus germ-free controls (six animals per group). Conflicting measurements for
repeated animal identifiers precluded a caecal comparison.

In c and d, bars show pointwise 95% bootstrap confidence intervals for the ratios
of group means. Green marks Holm-adjusted P < 0.05 within each family and blue marks
P ≥ 0.05. An open diamond indicates an undefined ratio interval. The germ-free
comparison did not test equivalence.

Source measurements from [Roje et al. (2024)](https://doi.org/10.1038/s41586-024-07754-w).
"""
    paths = []
    for stem, text in (("emd2_empirical", main), ("emd2_empirical_controls", controls)):
        path = Path(outdir) / f"{stem}_legend.md"
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


def make_figure(result, outdir: Path):
    """Export Figures 3 and S5 with their captions."""
    outdir = Path(outdir)
    paths = _save(create_figure(result), outdir, "emd2_empirical")
    paths.extend(_save(create_controls_figure(result), outdir, "emd2_empirical_controls"))
    paths.extend(write_legends(result, outdir))
    return paths
