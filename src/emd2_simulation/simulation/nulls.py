"""Host-only comparisons fitted to synthetic urinary BCPN.

Matching one urinary concentration leaves the underlying kinetics undetermined.
The caecum/plasma contrast depends on the assumed exchange and biliary rates.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from .model import Arm, Params, lumen_plasma_ratio, simulate, trace

HOST_ONLY = Arm("null", "Host-only null", gus=0.0, ox=0.0)

# Host generation, renal elimination and systemic loss can vary independently.
FREE = ["k_ox_host", "k_renal", "k_clear_p"]


def fit_host_only(p: Params, target_urine: float,
                  t: np.ndarray) -> tuple[Params, float]:
    """Fit three host rates to one urinary target and return the relative residual."""
    base = np.array([getattr(p, k) for k in FREE])

    def resid(logmul):
        q = replace(p, **{k: float(b * np.exp(m))
                          for k, b, m in zip(FREE, base, logmul)})
        try:
            tr = trace(simulate(q, HOST_ONLY, t), q)
        except Exception:
            # Penalise failed integrations relative to the magnitude of the target.
            return np.array([1e3 * max(abs(target_urine), 1.0)])
        return np.array([tr["c_BCPN_urine"][-1] - target_urine])

    sol = least_squares(resid, np.zeros(len(FREE)),
                        bounds=(-np.log(60.0) * np.ones(len(FREE)),
                                np.log(60.0) * np.ones(len(FREE))),
                        xtol=1e-14, ftol=1e-14)
    q = replace(p, **{k: float(b * np.exp(m))
                      for k, b, m in zip(FREE, base, sol.x)})
    tr = trace(simulate(q, HOST_ONLY, t), q)
    got = float(tr["c_BCPN_urine"][-1])
    rel_err = abs(got - target_urine) / max(abs(target_urine), 1e-30)
    return q, rel_err


def evaluate_null(p: Params, t: np.ndarray, segment: str = "cecum") -> dict:
    """Compare compartment outputs after matching simulated conventional urine."""
    from .conditions import BY_KEY

    conv = trace(simulate(p, BY_KEY["B"], t), p)
    target = float(conv["c_BCPN_urine"][-1])

    q, rel_err = fit_host_only(p, target, t)
    y_null = simulate(q, HOST_ONLY, t)
    tr_null = trace(y_null, q)

    return {
        "params": q,
        "urine_target": target,
        "urine_null": float(tr_null["c_BCPN_urine"][-1]),
        "urine_rel_err": rel_err,
        "ratio": lumen_plasma_ratio(y_null, q, segment),
        "cecum_null": float(tr_null[f"c_BCPN_{segment}"][-1]),
        "cecum_mech": float(conv[f"c_BCPN_{segment}"][-1]),
        "auc_null": float(tr_null["AUC_uro"][-1]),
        "auc_mech": float(conv["AUC_uro"][-1]),
        "scaling": {k: getattr(q, k) / getattr(p, k) for k in FREE},
        "profile": {s: float(tr_null[f"c_BCPN_{s}"][-1])
                    for s in ("duodenum", "jejunum", "ileum",
                              "cecum", "colon", "rectum")},
    }


def biliary_escape(p: Params, t: np.ndarray, segment: str = "cecum",
                   grid=(0.0, 0.01, 0.0227, 0.05, 0.10, 0.30)) -> dict:
    """Refit synthetic urine while varying assumed biliary BCPN delivery.

    Clearance fractions use the refitted host-only plasma clearance.
    """
    from .conditions import BY_KEY

    conv = trace(simulate(p, BY_KEY["B"], t), p)
    target = float(conv["c_BCPN_urine"][-1])

    rows = []
    for kb in grid:
        base = replace(p, k_bile_BCPN=float(kb))
        q, rel_err = fit_host_only(base, target, t)
        y = simulate(q, HOST_ONLY, t)
        ratio = lumen_plasma_ratio(y, q, segment)
        # Report fractions of both refitted and nominal plasma clearance.
        frac = kb / (kb + q.k_renal + q.k_clear_p) if kb else 0.0
        frac_nom = kb / (kb + p.k_renal + p.k_clear_p) if kb else 0.0
        rows.append({"k_bile_BCPN": float(kb),
                     "frac_of_plasma_clearance": float(frac),
                     "frac_of_nominal_clearance": float(frac_nom),
                     "null_ratio": float(ratio),
                     "urine_rel_err": float(rel_err),
                     "inverts_the_statistic": bool(ratio > 1.0)})

    breaking = [r for r in rows if r["inverts_the_statistic"]]
    return {
        "rows": rows,
        "urine_target": target,
        "min_k_bile_BCPN_breaking_V7": (min(r["k_bile_BCPN"] for r in breaking)
                                        if breaking else None),
        "min_frac_breaking_V7": (min(r["frac_of_plasma_clearance"]
                                     for r in breaking) if breaking else None),
        "min_frac_nominal_breaking_V7": (min(r["frac_of_nominal_clearance"]
                                             for r in breaking)
                                         if breaking else None),
        "interpretation": (
            "The ratio depends on the assigned exchange and biliary rates. A host-only "
            "ratio above one is possible while matching simulated conventional urine."),
    }


def fit_host_only_single(p: Params, target_urine: float,
                         t: np.ndarray, knob: str = "k_ox_host") -> Params:
    """Fit a single host rate to a synthetic urinary target, holding the others fixed."""
    base = getattr(p, knob)

    def resid(logmul):
        q = replace(p, **{knob: float(base * np.exp(logmul[0]))})
        try:
            tr = trace(simulate(q, HOST_ONLY, t), q)
        except Exception:
            return np.array([1e3 * max(abs(target_urine), 1.0)])
        return np.array([tr["c_BCPN_urine"][-1] - target_urine])

    sol = least_squares(resid, [0.0], bounds=([-np.log(200.0)], [np.log(200.0)]),
                        xtol=1e-14, ftol=1e-14)
    return replace(p, **{knob: float(base * np.exp(sol.x[0]))})


def host_parameter_cost(p: Params, t: np.ndarray) -> dict:
    """Compare the host oxidation rates that match each arm's simulated urine."""
    from .conditions import BY_KEY

    fits = {}
    for key in ("B", "BA"):
        tr = trace(simulate(p, BY_KEY[key], t), p)
        target = float(tr["c_BCPN_urine"][-1])
        q = fit_host_only_single(p, target, t, knob="k_ox_host")
        got = float(trace(simulate(q, HOST_ONLY, t), q)["c_BCPN_urine"][-1])
        fits[key] = {"params": q, "k_ox_host": q.k_ox_host,
                     "target": target, "achieved": got,
                     "rel_err": abs(got - target) / max(target, 1e-12)}

    fold = fits["B"]["k_ox_host"] / fits["BA"]["k_ox_host"]
    return {
        "fits": fits,
        "k_ox_fold_required": float(fold),
        "microbial_host_change": 1.0,
    }
