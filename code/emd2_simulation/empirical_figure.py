"""Publication figures drawn from published measurements and their reanalysis."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

from .empirical import TISSUE_LABEL

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


def _forest(ax, rows, labels, xlabel):
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


def _acute(ax, result, tissue):
    for group, colour, marker in (("BBN", GREEN, "o"), ("ABX/BBN", GOLD, "s")):
        rows = sorted((r for r in result["acute"]["descriptive"] if r["group"] == group and r["tissue"] == tissue), key=lambda r: r["time_h"])
        for r in rows:
            offsets = np.linspace(-0.12, 0.12, len(r["values_uM"]))
            ax.scatter(r["time_h"] + offsets, r["values_uM"], s=11, c=colour,
                       marker=marker, alpha=0.55, edgecolors="none")
        ax.plot([r["time_h"] for r in rows], [r["mean_uM"] for r in rows],
                marker=marker, ms=3.5, color=colour, label=group, lw=1.2)
    ax.set_xticks([1, 3, 6, 9])
    ax.set_xlabel("Hours after oral BBN bolus")
    ax.set_ylabel("BCPN (source-table µM)")
    ax.set_ylim(bottom=-5)
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
        texts += ax.get_xticklabels() + ax.get_yticklabels() + list(ax.texts)
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
        fig.savefig(path, dpi=450, facecolor="white", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def make_figure(result, outdir: Path):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 2, figsize=(183/25.4, 195/25.4), dpi=160)
        fig.subplots_adjust(left=0.12, right=0.96, top=0.89, bottom=0.14, hspace=0.76, wspace=0.47)
        fig.text(0.065, 0.975, "EMD2: published microbial transformation and toxicokinetics", weight="bold", fontsize=10, color=INK, va="top")
        fig.text(0.065, 0.949, "Secondary analysis of Roje et al. (2024) • measured BCPN, not simulated concentrations", fontsize=7, color=MUTED, va="top")
        ax = axes[0, 0]
        donor_ids = sorted({r["donor"] for r in result["human_donors"]})
        lookup = {(r["donor"], r["oxygen"]): r["mean_nM"] for r in result["human_donors"]}
        for donor in donor_ids:
            y = [lookup[donor, o] for o in ("0%O2", "10%O2", "21%O2")]
            ax.plot(range(3), y, color="#aab8c6", lw=0.7, marker="o", ms=3)
        mean = [np.mean([lookup[k, o] for k in donor_ids]) for o in ("0%O2", "10%O2", "21%O2")]
        ax.plot(range(3), mean, color=GREEN, lw=2, marker="D", ms=4, label="Donor mean")
        ax.set_xticks(range(3), ["0%", "10%", "21%"])
        ax.set_xlabel("Culture oxygen at 24 h")
        ax.set_ylabel("Culture BCPN (nM)")
        ax.legend(frameon=False)
        _style(ax, "a", "Human communities · 10 donors")

        ax = axes[0, 1]
        row = next(r for r in result["consortium"] if r["Tissue"] == "CEC")
        for x, key, colour, label in ((0, "observations_b", BLUE, "2-member"), (1, "observations_a", GREEN, "3-member")):
            vals = [r["value"] for r in row[key]]
            ax.scatter(x + np.linspace(-0.13, 0.13, len(vals)), vals, s=24, c=colour, edgecolors="white", linewidths=0.4)
            ax.hlines(np.mean(vals), x-0.20, x+0.20, color=colour, lw=2)
        ax.set_xticks([0, 1], ["2-member", "3-member"])
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylabel("Caecal BCPN (µM)")
        ax.text(0.04, 0.94, f"8 animals/arm; Holm P = {row['p_holm']:.4f}", transform=ax.transAxes, fontsize=7, va="top")
        ax.set_ylim(0, 6.1)
        _style(ax, "b", "Adding a converting isolate")

        ax = axes[1, 0]
        order = ("SI1", "SI2", "SI3", "CEC", "COL", "REC", "PL", "LVR", "KID")
        rows = [next(r for r in result["consortium"] if r["Tissue"] == k) for k in order]
        labels = ["SI1", "SI2", "SI3", "Caecum", "Colon", "Rectum", "Plasma", "Liver", "Kidney"]
        _forest(ax, rows, labels, "BCPN mean ratio: 3-member / 2-member")
        ax.set_xlim(0.08, 12)
        ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
        _style(ax, "c", "Contrasts within each compartment")

        ax = axes[1, 1]
        _acute(ax, result, "Cecum")
        _style(ax, "d", "Acute dose · caecum†")
        ax = axes[2, 0]
        _acute(ax, result, "PL")
        _style(ax, "e", "Acute dose · plasma†")

        ax = axes[2, 1]
        order = [("3wk", "3wk_1213"), ("3wk", "Neg_12wk_0629"), ("6wk", "Neg_12wk_0629"),
                 ("7wk", "Neg_7wk_0211"), ("12wk", "Neg_12wk_0629")]
        rows = [next(r for r in result["urine"] if (r["Length"], r["Batch"]) == key) for key in order]
        _forest(ax, rows, ["3 wk · 1213", "3 wk · 0629", "6 wk · 0629", "7 wk · 0211", "12 wk · 0629"], "24-h urinary amount: BBN / ABX–BBN")
        ax.set_xlim(0.2, 50)
        ax.set_xticks([0.2, 1, 5, 25], ["0.2", "1", "5", "25"])
        _style(ax, "f", "Urinary elimination by batch")
        fig.text(0.065, 0.060, "c, f: ratios compare treatment groups within a matrix, not lumen with plasma. Lines: pointwise 95% bootstrap CIs.", fontsize=6.4, color=MUTED)
        fig.text(0.065, 0.043, "Green: Holm P < 0.05 within the stated family. Open diamonds: ratio CI undefined because of zero denominators.", fontsize=6.4, color=MUTED)
        fig.text(0.065, 0.026, "† Acute data are descriptive: ID-prefix mapping, counts and published statistics need reconciliation. No temporal source attribution.", fontsize=6.4, color=MUTED)
        paths = _save(fig, outdir, "figure2d_emd2_simulation")

        fig, axes = plt.subplots(2, 2, figsize=(183/25.4, 145/25.4), dpi=160)
        fig.subplots_adjust(left=0.13, right=0.96, top=0.87, bottom=0.13, hspace=0.62, wspace=0.46)
        fig.text(0.065, 0.965, "EMD2: controls, exposure heterogeneity and isolate activity", fontsize=10, weight="bold", color=INK)
        ax = axes[0, 0]
        rows = result["germ_free_controls"]
        _forest(ax, rows, [TISSUE_LABEL[r["Tissue"]] for r in rows], "Mean BCPN: GF + ABX / GF")
        ax.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
        _style(ax, "a", "Germ-free antibiotic controls")
        ax = axes[0, 1]
        rows = result["bladder"]
        _forest(ax, rows, [r["Length"].replace("wk", " wk") + " · " + r["Batch"].split("_")[-1] for r in rows], "Mean bladder BCPN: BBN / ABX–BBN")
        _style(ax, "b", "Bladder measurements by batch")
        ax = axes[1, 0]
        _acute(ax, result, "Liver")
        _style(ax, "c", "Acute dose · liver (descriptive)")
        ax = axes[1, 1]
        signals = [r for r in result["isolate_signals"] if r["time_h"] == 12 and r["strain"] == "Escherichia coli ED1a"]
        for x, oxygen in enumerate(("0%O2", "10%O2", "21%O2")):
            row = next((r for r in signals if r["oxygen"] == oxygen), None)
            if row:
                ax.scatter(x + np.linspace(-0.1, 0.1, len(row["signals"])), np.asarray(row["signals"])/1e3, s=20, c=GREEN)
                ax.hlines(row["mean_signal"]/1e3, x-0.17, x+0.17, color=GREEN, lw=1.5)
        ax.set_xticks(range(3), ["0%", "10%", "21%"])
        ax.set_xlabel("Culture oxygen · 12 h")
        ax.set_ylabel("BCPN signal (10³ a.u.)")
        _style(ax, "d", "E. coli ED1a · technical replicates")
        fig.text(0.065, 0.055, "a: 6 animals/arm; caecum excluded because duplicate IDs have conflicting, unlabelled readings. No equivalence test.", fontsize=6.4, color=MUTED)
        fig.text(0.065, 0.035, "b: all collection strata retained. c: count/statistic discrepancies flagged. d: signal is not a concentration or biological n.", fontsize=6.4, color=MUTED)
        paths += _save(fig, outdir, "figureS_emd2_empirical_controls")
        return paths
