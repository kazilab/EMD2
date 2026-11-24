"""Sensitivity to independent log-normal perturbations of assigned parameters.

Each draw shares the same parameters across arms. Segment profiles, source
tracking and arm capacities remain fixed. The two-member capacity and biliary
delivery are examined in separate scenarios.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .model import Arm, Params, lumen_plasma_ratio, simulate, trace

# Assigned log-scale standard deviations of the parameter multipliers.
FREE_SIGMA = {
    "V_plasma": 0.15, "V_bladder": 0.15, "V_kidney": 0.15,
    "dose_rate": 0.15, "Q_bile": 0.25,
    "ka_BBN": 0.25, "k_uptake": 0.25, "k_gluc": 0.25, "k_bile": 0.25,
    "k_renal": 0.25, "k_kidney_out": 0.25, "k_void": 0.25,
    "k_clear_p": 0.25, "k_clear_BBN": 0.25,
    "k_ox_host": 0.35,
    "ka_BCPN": 0.30, "k_back": 0.30,
    "Vmax_gus": 0.40, "Km_gus": 0.40, "Vmax_ox": 0.40, "Km_ox": 0.40,
    "k_uro": 0.35,
}

DEFAULT_N = 400
DEFAULT_SEED = 20260825


def sample_params(rng: np.random.Generator, base: Params | None = None) -> Params:
    """One draw: independent log-normal multipliers on the free rate constants."""
    base = base or Params()
    draw = {k: float(getattr(base, k) * np.exp(rng.normal(0.0, s)))
            for k, s in FREE_SIGMA.items()}
    return replace(base, **draw)


def _observe(p: Params, arm: Arm, t: np.ndarray, ratio_segment: str) -> dict:
    y = simulate(p, arm, t)
    tr = trace(y, p)
    return {
        "AUC_uro": float(tr["AUC_uro"][-1]),
        "AUC_uro_micro": float(tr["AUC_uro_micro"][-1]),
        "c_BCPN_urine": float(tr["c_BCPN_urine"][-1]),
        "c_BCPN_cecum": float(tr["c_BCPN_cecum"][-1]),
        "c_BCPN_plasma": float(tr["c_BCPN_plasma"][-1]),
        "c_BCPN_kidney": float(tr["c_BCPN_kidney"][-1]),
        "micro_fraction": float(tr["micro_fraction"][-1]),
        "ratio_cecum": lumen_plasma_ratio(y, p, ratio_segment),
    }


def run_ensemble(arms, t: np.ndarray, ratio_segment: str = "cecum",
                 n: int = DEFAULT_N, seed: int = DEFAULT_SEED,
                 base: Params | None = None, robustness=None, progress=None) -> dict:
    """Return observable percentiles and check frequencies across parameter draws."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng(seed)
    base = base or Params()
    keys = [a.key for a in arms]

    draws: list[dict] = []
    n_failed = 0
    for draw_index in range(n):
        q = sample_params(rng, base)
        try:
            draws.append({a.key: _observe(q, a, t, ratio_segment) for a in arms})
        except Exception:
            n_failed += 1
        if progress is not None and ((draw_index + 1) % max(1, n // 10) == 0 or draw_index + 1 == n):
            progress(f"Sensitivity ensemble: {draw_index + 1}/{n} draws attempted")

    if not draws:
        raise RuntimeError("no ensemble draw integrated successfully")

    # Form arm ratios within each draw to retain their shared parameter variation.
    ref = keys[0] if "B" not in keys else "B"
    for d in draws:
        ref_auc = d[ref]["AUC_uro"]
        for k in keys:
            d[k]["AUC_rel_conventional"] = (d[k]["AUC_uro"] / ref_auc
                                            if ref_auc > 0 else float("nan"))

    obs = list(draws[0][keys[0]].keys())
    bands = {k: {o: dict(zip(("p5", "p50", "p95"), (
        float(v) for v in np.percentile([d[k][o] for d in draws], [5, 50, 95]))))
        for o in obs} for k in keys}

    if robustness is None:
        from ..analysis.checks import ROBUSTNESS
        robustness = ROBUSTNESS
    audit = {}
    for cid, fn in robustness.items():
        hits = [bool(fn(d)) for d in draws]
        audit[cid] = float(np.mean(hits))

    auc_2mem_over_b = np.array([d["2Mem"]["AUC_uro"] / d["B"]["AUC_uro"]
                                for d in draws]) if "2Mem" in keys else np.array([])
    return {
        "n_requested": n, "n_ok": len(draws), "n_failed": n_failed,
        "seed": seed, "bands": bands, "audit": audit,
        "auc_2mem_over_b": dict(zip(("p5", "p50", "p95"), (
            float(v) for v in np.percentile(auc_2mem_over_b, [5, 50, 95]))))
        if len(auc_2mem_over_b) else None,
        "interpretation": (
            "Percentiles describe sensitivity to the assigned parameter ranges. "
            "Model structure and arm capacities are fixed across draws."),
    }

