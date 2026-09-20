"""Conditional synthetic recoverability for measurement sets informed by MTBLS3581.

Synthetic observations are generated at nominal kinetics. Vmax_ox is profiled
while Vmax_gus is refitted; all other parameters are fixed. The objective uses
terminal concentrations and a 15% relative-residual threshold, not observed
animal values, sample sizes, collection times or covariance. Results compare
measurement sets under these assumptions, not empirical identifiability.

The sample table counts LC-MS acquisitions. Filename parsing cannot establish
independent biological n, and some anatomy labels conflict with filenames.
Record counts are retained as deposited; metadata_conflicts flags a known
conflict without silently relabelling anatomy. No animal n is inferred.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .model import Arm, Params, simulate, trace

DATA = Path(__file__).parent / "data"

# The `Abx treatment` column also carries spike/recovery samples labelled with
# nitrosamine names (DBN, PBN, EHBN and their acid analogues). They are analytical
# standards run in matrix, not experimental arms, and counting them as arms makes
# the design look better covered than it is.
REAL_ARMS = {"B", "BA", "Germ-Free", "Monocolonized", "2Mem", "3Mem"}




def load_design() -> dict:
    """Analytical-record counts as {arm: {compartment: {week: n}}}.

    These are acquisitions, not animals. No independent biological n is inferred.
    """
    path = DATA / "s_MTBLS3581.txt"
    rows = list(csv.reader(path.open(), delimiter="\t"))[1:]
    out: dict = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r[10] != "experimental sample":
            continue
        arm, part, wk = r[18], r[4], r[21]
        out[arm][part][wk] = out[arm][part].get(wk, 0) + 1
    return {a: {p: dict(w) for p, w in d.items()} for a, d in out.items()}


def metadata_conflicts() -> list[dict]:
    """Flag explicit liver-filename/colon-label conflicts without correcting them.

    This is a targeted audit of a known inconsistency, not a complete specimen
    manifest. The original metadata and all record counts remain unchanged.
    """
    with (DATA / "s_MTBLS3581.txt").open() as stream:
        rows = list(csv.reader(stream, delimiter="\t"))[1:]
    return [{"source_name": r[0], "arm": r[18], "week": r[21],
             "recorded_compartment": r[4], "filename_compartment": "liver"}
            for r in rows if r[10] == "experimental sample"
            and r[18] in REAL_ARMS and r[4].casefold() == "colon"
            and re.search(r"_LVR(?:_|\.)", r[0], re.I)]


def coverage(part: str, design: dict | None = None) -> dict:
    """Acquisition counts and arm coverage; not independent biological n."""
    design = design if design is not None else load_design()
    counts = {a: sum(sum(v.values()) for key, v in design.get(a, {}).items()
                     if key.casefold() == part.casefold()) for a in REAL_ARMS}
    arms = sorted(a for a, n in counts.items() if n)
    return {"compartment": part, "records": sum(counts.values()),
            "arms": arms, "n_arms": len(arms), "count_unit": "LC-MS acquisition",
            "biological_n_verified": False}


def design_facts(design: dict) -> dict:
    """The structural gaps, computed rather than asserted."""
    def arms_with(part: str) -> set:
        # Case-insensitive on purpose. The deposition writes every compartment
        # in lower case EXCEPT `Plasma`, so an exact-match lookup for "plasma"
        # silently returns the empty set -- a wrong answer that looks like a
        # finding about the design rather than a string-matching slip.
        want = part.casefold()
        return {a for a, d in design.items() if a in REAL_ARMS
                and any(k.casefold() == want and sum(v.values())
                        for k, v in d.items())}

    urine = arms_with("urine")
    bile = arms_with("bile")
    cecum = arms_with("cecum")
    return {
        "arms_with_urine": sorted(urine),
        "arms_with_bile": sorted(bile),
        "arms_with_cecum": sorted(cecum),
        "arms_with_plasma": sorted(arms_with("plasma")),
        "arms_with_kidney": sorted(arms_with("kidney")),
        "coverage": {pt: coverage(pt, design) for pt in sorted(
            {p for a, d in design.items() if a in REAL_ARMS for p in d})},
        "metadata_conflicts": metadata_conflicts(),
        "biological_n_verified": False,
        "urine_bile_disjoint": len(urine & bile) == 0,
        # REAL_ARMS, not len(design): the `Abx treatment` column also carries
        # analytical standards (DBN/PBN/EHBN and acid analogues) and blanks, and
        # counting those as arms reported 13 experimental arms where there are 6.
        "n_arms_total": len({a for a in design if a in REAL_ARMS}),
        "non_arm_labels": sorted(a for a in design if a not in REAL_ARMS),
    }


# Endpoint types actually loaded by this build. This is an implementation
# inventory, not an inference about assay availability from anatomy labels.
IMPORTED_ENDPOINTS = ("metabolite_concentrations", "urinary_metabolite_amounts")
FUNCTIONAL_ENDPOINTS = {
    "KCC6": "chronic inflammation",
    "KCC7": "immunosuppression",
    "KCC8": "receptor-mediated effects",
    "KCC2": "genotoxicity",
    "KCC10": "altered cell proliferation, cell death or nutrient supply",
}


def host_response_verdict(design: dict) -> dict:
    """Declare endpoints absent from the current imported/modelled data."""
    parts = sorted({part for arm, d in design.items() if arm in REAL_ARMS
                    for part in d})
    return {
        "compartments": parts,
        "imported_endpoints": list(IMPORTED_ENDPOINTS),
        "functional_endpoints_required": dict(FUNCTIONAL_ENDPOINTS),
        "host_response_unidentifiable": True,
        "genotoxicity_unidentifiable": True,
        "kcc10_response_unidentifiable": True,
        "unconstrainable_edges": list(FUNCTIONAL_ENDPOINTS),
        "basis": ("No corresponding functional response endpoint is imported "
                  "or modelled in this build. Compartment names do not establish "
                  "assay absence from the deposition or source study."),
    }


def _observables(p: Params, arm: Arm, t: np.ndarray, compartments: list[str]) -> np.ndarray:
    tr = trace(simulate(p, arm, t), p)
    vals = []
    for c in compartments:
        key = "c_BCPN_urine" if c == "urine" else f"c_BCPN_{c}"
        vals.append(float(tr[key][-1]))
    return np.asarray(vals)


def _informative(p: Params, t: np.ndarray, compartments: list[str],
                 arm: Arm, grid: np.ndarray, tol: float = 1e-6) -> bool:
    """Does this arm's measurement respond to Vmax_ox AT ALL?

    An arm with no microbial oxidase (germ-free, or the two-member consortium)
    produces the same luminal BCPN for every value of the parameter being
    profiled -- identical to machine precision. Its residuals are therefore
    structurally zero, and averaging them into an RMSE makes the parameter look
    LESS identifiable the more such arms are included. That is an artefact of
    the metric, not a property of the design, so uninformative arms are detected
    and reported rather than silently diluting the score.
    """
    lo = _observables(replace(p, Vmax_ox=float(grid[0])), arm, t, compartments)
    hi = _observables(replace(p, Vmax_ox=float(grid[-1])), arm, t, compartments)
    scale = np.maximum(np.abs(lo) + np.abs(hi), 1e-12)
    return bool(np.max(np.abs(hi - lo) / scale) > tol)


def profile_capacity(p: Params, t: np.ndarray, compartments: list[str],
                     arms: list[Arm], grid: np.ndarray | None = None,
                     rel_noise: float = 0.15) -> dict:
    """Profile microbial oxidation capacity against a given measurement set.

    ``compartments`` is what the experiment measured; ``arms`` is which arms it
    is represented in. Vmax_gus is refitted at each fixed Vmax_ox against
    synthetic terminal values; this is conditional recoverability.

    ``rel_noise`` is 15% relative error on a metabolite concentration, which is
    optimistic for LC-MS across tissue types.

    The admissible set is scored on the WORST relative residual, not the mean.
    A measurement set identifies a parameter only if EVERY observable it
    contains is reproduced within error; a set that misses one compartment
    badly while matching four others has not identified anything, and averaging
    lets exactly that pass. Reporting the mean as well makes the difference
    visible instead of hiding the choice.
    """
    if grid is None:
        grid = np.linspace(0.05, 2.5, 18) * p.Vmax_ox

    informative = [a for a in arms if _informative(p, t, compartments, a, grid)]
    truth = np.concatenate([_observables(p, a, t, compartments) for a in arms])
    scale = np.maximum(np.abs(truth), 1e-9)

    rmse = np.zeros_like(grid)
    worst = np.zeros_like(grid)
    for i, vox in enumerate(grid):
        def resid(v):
            q = replace(p, Vmax_ox=float(vox), Vmax_gus=abs(float(v[0])))
            try:
                pred = np.concatenate([_observables(q, a, t, compartments)
                                       for a in arms])
            except Exception:
                return np.full_like(truth, 1e3)
            return (pred - truth) / scale

        # Bounds are MULTIPLES of nominal, never absolute numbers. An earlier
        # version hard-coded [1e-4, 500], which silently encoded the old
        # 1000-fold-low dose scale and threw "initial guess outside bounds" the
        # moment the dose was put on a real mass balance.
        sol = least_squares(resid, [p.Vmax_gus],
                            bounds=([1e-5 * p.Vmax_gus], [100.0 * p.Vmax_gus]),
                            xtol=1e-12, ftol=1e-12)
        rmse[i] = float(np.sqrt(np.mean(sol.fun ** 2)))
        worst[i] = float(np.max(np.abs(sol.fun)))

    mask = worst <= rel_noise
    admissible = grid[mask]
    admissible_mean = grid[rmse <= rel_noise]
    width = (float(admissible.max() - admissible.min()) / p.Vmax_ox
             if len(admissible) else 0.0)
    # `width` is max-min, so it is only a ridge width if the admissible set is
    # a single interval. A split set would report the gap as though it were
    # admissible. Contiguous in every set tested here, but asserted rather than
    # assumed, because the failure is silent and flatters the design.
    contiguous = bool(len(admissible) <= 1
                      or np.all(np.diff(np.flatnonzero(mask)) == 1))
    # A ridge that runs off the end of the tested grid is not "wide", it is
    # UNBOUNDED within the range examined, and quoting its width as a percentage
    # of nominal reports a property of the grid as though it were a property of
    # the design.
    saturated = bool(len(admissible)
                     and (admissible.min() <= grid[0] + 1e-12
                          or admissible.max() >= grid[-1] - 1e-12))
    return {"grid": grid, "rmse": rmse, "worst": worst,
            "admissible": admissible,
            "ridge_width": width,
            "ridge_width_mean_metric": (
                float(admissible_mean.max() - admissible_mean.min()) / p.Vmax_ox
                if len(admissible_mean) else 0.0),
            "saturated": saturated,
            "contiguous": contiguous,
            "n_arms": len(arms), "n_informative_arms": len(informative),
            "uninformative_arms": [a.key for a in arms if a not in informative],
            # Named for what it tests. The grid floor is 0.05 x nominal, not
            # zero, so the old name `zero_admissible` claimed the parameter
            # could be zero when it only said the ridge reached the bottom of
            # the tested range -- the lower half of `saturated`.
            "admissible_at_grid_floor": bool(len(admissible)
                                             and admissible.min() <= grid[0])}


# The measurement sets worth comparing. The last two exist because of a fact the
# earlier version of this module never used: urine is deposited in only two arms,
# but KIDNEY is deposited in all six (27/26/12/12/16/16). The elimination limb
# does have cross-arm coverage -- just not at the compartment the BBN literature
# habitually reports.
SETS = {
    "urine_only":        (["urine"], ("B", "BA")),
    "urine_plus_lumen":  (["urine", "cecum", "colon"], ("B", "BA")),
    "lumen_all_arms":    (["cecum", "colon"], ("B", "BA", "Germ-Free")),
    "kidney_all_arms":   (["kidney"],
                          ("B", "BA", "Germ-Free", "Monocolonized", "2Mem", "3Mem")),
    "lumen_plus_kidney": (["cecum", "colon", "kidney"],
                          ("B", "BA", "Germ-Free", "Monocolonized", "2Mem", "3Mem")),
}

SET_LABELS = {
    "urine_only": "urine only (B, BA)",
    "urine_plus_lumen": "urine + caecum + colon (B, BA)",
    "lumen_all_arms": "lumen, three arms",
    "kidney_all_arms": "kidney, all six arms",
    "lumen_plus_kidney": "lumen + kidney, all six arms",
}


def run(p: Params, t: np.ndarray) -> dict:
    """Compare synthetic measurement sets under fixed nuisance kinetics."""
    from .conditions import BY_KEY

    design = load_design()
    facts = design_facts(design)

    out = {"design": design, "facts": facts}
    for name, (compartments, arm_keys) in SETS.items():
        out[name] = profile_capacity(p, t, list(compartments),
                                     [BY_KEY[k] for k in arm_keys])

    # "degenerate" means the microbial capacity is not pinned: the admissible
    # ridge runs off the tested grid, so urine alone bounds the parameter
    # nowhere within a 50-fold range.
    out["urine_only_degenerate"] = bool(
        out["urine_only"]["saturated"]
        or out["urine_only"]["admissible_at_grid_floor"])
    return out
