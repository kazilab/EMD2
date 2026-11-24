"""Conditional parameter profiles for synthetic terminal concentrations.

The profile varies Vmax_ox and refits Vmax_gus with the remaining kinetics
fixed. The default admissibility criterion allows up to 15% relative error
in every observable. MTBLS3581 supplies compartment coverage by arm.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from ..simulation.model import Arm, Params, simulate, trace

from ..processing.store import load_acquisitions

# Treatment labels also include analytical standards and blanks.
REAL_ARMS = {"B", "BA", "Germ-Free", "Monocolonized", "2Mem", "3Mem"}


def load_design() -> dict:
    """Count acquisitions by arm, compartment and week."""
    rows = load_acquisitions()
    out: dict = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["sample_type"] != "experimental sample":
            continue
        arm, part, wk = row["arm"], row["compartment"], row["week"]
        out[arm][part][wk] = out[arm][part].get(wk, 0) + 1
    return {a: {p: dict(w) for p, w in d.items()} for a, d in out.items()}


def metadata_conflicts() -> list[dict]:
    """Return acquisitions labelled colon whose filenames identify liver."""
    rows = load_acquisitions()
    return [{"source_name": row["source_name"], "arm": row["arm"], "week": row["week"],
             "recorded_compartment": row["compartment"], "filename_compartment": "liver"}
            for row in rows if row["sample_type"] == "experimental sample"
            and row["arm"] in REAL_ARMS and row["compartment"].casefold() == "colon"
            and re.search(r"_LVR(?:_|\.)", row["source_name"], re.I)]


def coverage(part: str, design: dict | None = None) -> dict:
    """Summarise acquisition coverage across experimental arms."""
    design = design if design is not None else load_design()
    counts = {a: sum(sum(v.values()) for key, v in design.get(a, {}).items()
                     if key.casefold() == part.casefold()) for a in REAL_ARMS}
    arms = sorted(a for a, n in counts.items() if n)
    return {"compartment": part, "records": sum(counts.values()),
            "arms": arms, "n_arms": len(arms), "count_unit": "LC-MS acquisition",
            "biological_n_verified": False}


def design_facts(design: dict) -> dict:
    """Return observed compartment coverage and anatomy-label conflicts."""
    def arms_with(part: str) -> set:
        # MTBLS3581 uses mixed case for compartment labels.
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
        "n_arms_total": len({a for a in design if a in REAL_ARMS}),
        "non_arm_labels": sorted(a for a in design if a not in REAL_ARMS),
    }


# Endpoints represented by the compartment model.
IMPORTED_ENDPOINTS = ("metabolite_concentrations", "urinary_metabolite_amounts")
FUNCTIONAL_ENDPOINTS = {
    "KCC6": "chronic inflammation",
    "KCC7": "immunosuppression",
    "KCC8": "receptor-mediated effects",
    "KCC2": "genotoxicity",
    "KCC10": "altered cell proliferation, cell death or nutrient supply",
}


def host_response_verdict(design: dict) -> dict:
    """Describe which functional endpoints the model can assess."""
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
        "basis": ("The model includes metabolite measurements and no functional "
                  "response measurements. Compartment labels alone cannot establish "
                  "assay absence in the source study."),
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
    """Check whether terminal observations change between the two grid limits."""
    lo = _observables(replace(p, Vmax_ox=float(grid[0])), arm, t, compartments)
    hi = _observables(replace(p, Vmax_ox=float(grid[-1])), arm, t, compartments)
    scale = np.maximum(np.abs(lo) + np.abs(hi), 1e-12)
    return bool(np.max(np.abs(hi - lo) / scale) > tol)


def profile_capacity(p: Params, t: np.ndarray, compartments: list[str],
                     arms: list[Arm], grid: np.ndarray | None = None,
                     rel_noise: float = 0.15, progress=None) -> dict:
    """Refit Vmax_gus at each Vmax_ox using synthetic terminal concentrations.

    A grid point is admissible when every relative residual is within rel_noise.
    The root-mean-square residual is also reported.
    """
    if grid is None:
        grid = np.linspace(0.05, 2.5, 18) * p.Vmax_ox

    informative = [a for a in arms if _informative(p, t, compartments, a, grid)]
    truth = np.concatenate([_observables(p, a, t, compartments) for a in arms])
    scale = np.maximum(np.abs(truth), 1e-9)

    rmse = np.zeros_like(grid)
    worst = np.zeros_like(grid)
    for i, vox in enumerate(grid):
        if progress is not None:
            progress(f"Synthetic profile: grid point {i + 1}/{len(grid)}")
        def resid(v):
            q = replace(p, Vmax_ox=float(vox), Vmax_gus=abs(float(v[0])))
            try:
                pred = np.concatenate([_observables(q, a, t, compartments)
                                       for a in arms])
            except Exception:
                return np.full_like(truth, 1e3)
            return (pred - truth) / scale

        # Express optimisation bounds relative to the configured capacity.
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
    # A single width covers the admissible range only when it is contiguous.
    contiguous = bool(len(admissible) <= 1
                      or np.all(np.diff(np.flatnonzero(mask)) == 1))
    # A profile touching either boundary may extend beyond the tested grid.
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
            "admissible_at_grid_floor": bool(len(admissible)
                                             and admissible.min() <= grid[0])}


# Synthetic profiles use urine in two arms and kidney in all six.
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


def run(p: Params, t: np.ndarray, progress=None) -> dict:
    """Compare synthetic measurement sets under fixed nuisance kinetics."""
    from ..simulation.conditions import BY_KEY

    design = load_design()
    facts = design_facts(design)

    out = {"design": design, "facts": facts}
    for name, (compartments, arm_keys) in SETS.items():
        if progress is not None:
            progress(f"Conditional synthetic profile: {name}")
        out[name] = profile_capacity(p, t, list(compartments),
                                     [BY_KEY[k] for k in arm_keys], progress=progress)

    # Boundary contact leaves the urine-only range incomplete.
    out["urine_only_degenerate"] = bool(
        out["urine_only"]["saturated"]
        or out["urine_only"]["admissible_at_grid_floor"])
    return out
