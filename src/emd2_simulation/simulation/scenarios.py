"""Explicit numerical sensitivity scenarios under assigned kinetics."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from .model import Arm, Params, simulate, trace, lumen_plasma_ratio
from .ensemble import sample_params, DEFAULT_SEED


def null_cost_subsample(t: np.ndarray, n: int = 40, seed: int = DEFAULT_SEED + 1,
                        base: Params | None = None, progress=None) -> dict:
    """Estimate the host oxidation contrast by refitting each parameter draw."""
    from .nulls import host_parameter_cost

    rng = np.random.default_rng(seed)
    base = base or Params()
    folds = []
    for draw_index in range(n):
        q = sample_params(rng, base)
        try:
            folds.append(float(host_parameter_cost(q, t)["k_ox_fold_required"]))
        except Exception:
            continue
        if progress is not None:
            progress(f"Null-cost audit: {draw_index + 1}/{n} draws attempted")
    if not folds:
        raise RuntimeError("no null-cost draw completed")
    a = np.asarray(folds)
    return {"n_ok": len(folds), "n_requested": n, "seed": seed,
            "fold": dict(zip(("p5", "p50", "p95"),
                             (float(v) for v in np.percentile(a, [5, 50, 95])))),
            "frac_above_1p5": float(np.mean(a > 1.5))}


def capacity_decomposition(p: Params, t: np.ndarray,
                           ratio_segment: str = "cecum") -> dict:
    """Compare endpoint concentrations and exposure across the four combinations
    of assumed deconjugation and oxidation capacity.
    """
    cells = {}
    for gus, ox in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
        y = simulate(p, Arm("x", "x", gus=gus, ox=ox), t)
        tr = trace(y, p)
        cells[f"gus{gus:.2f}_ox{ox:.2f}"] = {
            "gus": gus, "ox": ox,
            "c_BCPN_cecum": float(tr["c_BCPN_cecum"][-1]),
            "c_BCPN_plasma": float(tr["c_BCPN_plasma"][-1]),
            "c_BCPN_urine": float(tr["c_BCPN_urine"][-1]),
            "AUC_uro": float(tr["AUC_uro"][-1]),
            "AUC_uro_host": float(tr["AUC_uro_host"][-1]),
            "AUC_uro_micro": float(tr["AUC_uro_micro"][-1]),
            "ratio_cecum": lumen_plasma_ratio(y, p, ratio_segment),
        }

    # V2 compares both urinary concentration and cumulative exposure.
    full = cells["gus1.00_ox1.00"]
    v2 = {}
    for lab, (gus, ox) in (("deplete_both", (0.05, 0.05)),
                           ("deplete_gus_only", (0.05, 1.00)),
                           ("deplete_ox_only", (1.00, 0.05))):
        tr = trace(simulate(p, Arm("x", "x", gus=gus, ox=ox), t), p)
        v2[lab] = {
            "gus": gus, "ox": ox,
            "c_BCPN_urine": float(tr["c_BCPN_urine"][-1]),
            "AUC_uro": float(tr["AUC_uro"][-1]),
            "v2_holds": bool(tr["c_BCPN_urine"][-1] < 0.95 * full["c_BCPN_urine"]
                             and tr["AUC_uro"][-1] < 0.95 * full["AUC_uro"]),
        }

    gus_only = cells["gus1.00_ox0.00"]["AUC_uro"]
    return {
        "cells": cells,
        "v2_attribution": v2,
        "oxidase_effect_on_total_AUC": float(full["AUC_uro"] / gus_only),
        "oxidase_is_a_systemic_sink": bool(full["AUC_uro"] < gus_only),
        "interpretation": (
            "At the reference settings, deconjugation accounts for the systemic "
            "decrease under microbial depletion. Adding oxidation increases luminal "
            "BCPN and lowers the total exposure index relative to deconjugation alone."),
    }


def dose_only_sensitivity(p: Params, t: np.ndarray,
                          multipliers=(0.7, 1.0, 1.3),
                          ratio_segment: str = "cecum") -> dict:
    """Compare dose-only changes with joint rescaling of dose, Vmax and Km."""
    from .conditions import BY_KEY

    dose_only, joint = [], []
    for m in multipliers:
        q = replace(p, dose_rate=p.dose_rate * m)
        y = simulate(q, BY_KEY["B"], t)
        tr = trace(y, q)
        dose_only.append({
            "multiplier": float(m),
            "dose_rate": float(q.dose_rate),
            "ratio_cecum": lumen_plasma_ratio(y, q, ratio_segment),
            "c_BCPN_cecum": float(tr["c_BCPN_cecum"][-1]),
            "micro_fraction": float(tr["micro_fraction"][-1]),
        })
        r = replace(p, dose_rate=p.dose_rate * m,
                    Vmax_gus=p.Vmax_gus * m, Km_gus=p.Km_gus * m,
                    Vmax_ox=p.Vmax_ox * m, Km_ox=p.Km_ox * m)
        joint.append({"multiplier": float(m),
                      "ratio_cecum": lumen_plasma_ratio(
                          simulate(r, BY_KEY["B"], t), r, ratio_segment)})

    ref = next(d["ratio_cecum"] for d in dose_only if d["multiplier"] == 1.0)
    spread = max(abs(d["ratio_cecum"] - ref) / ref for d in dose_only)
    jref = next(d["ratio_cecum"] for d in joint if d["multiplier"] == 1.0)
    jspread = max(abs(d["ratio_cecum"] - jref) / jref for d in joint)
    return {
        "dose_only": dose_only, "joint_rescaling": joint,
        "max_rel_change_dose_only": float(spread),
        "max_rel_change_joint": float(jspread),
        "interpretation": (
            "Jointly rescaling dose, Vmax and Km preserves the caecum/plasma ratio. "
            "Dose-only changes can alter the ratio as substrate saturation changes."),
    }


def two_member_gus_sensitivity(p: Params, t: np.ndarray,
                               grid=(0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00),
                               ratio_segment: str = "cecum") -> dict:
    """Evaluate two-member exposure over the specified deconjugation capacities."""
    base_auc = float(trace(simulate(p, Arm("B", "B", 1.0, 1.0), t), p)["AUC_uro"][-1])
    gf_auc = float(trace(simulate(p, Arm("GF", "GF", 0.0, 0.0), t), p)["AUC_uro"][-1])
    abx_auc = float(trace(simulate(p, Arm("BA", "BA", 0.05, 0.05), t), p)["AUC_uro"][-1])

    rows = []
    for g in grid:
        tr = trace(simulate(p, Arm("2M", "2M", gus=float(g), ox=0.0), t), p)
        auc = float(tr["AUC_uro"][-1])
        rows.append({"gus": float(g), "AUC_uro": auc,
                     "frac_of_conventional": auc / base_auc,
                     "exceeds_abx": bool(auc > abx_auc),
                     "exceeds_conventional_95": bool(auc >= 0.95 * base_auc),
                     "exceeds_gf_1p5": bool(auc > 1.5 * gf_auc)})
    passing = [r["gus"] for r in rows if r["exceeds_abx"]]
    return {"rows": rows, "conventional_AUC": base_auc, "abx_AUC": abx_auc,
            "gf_AUC": gf_auc,
            "min_gus_beating_abx": min(passing) if passing else None}
