"""Uncertainty propagation for the EMD2 build.

Every magnitude in this model is illustrative (`fit/PROVENANCE.md`: the MAF
identifies BBN and BCPN but carries no values). Reporting the arm table as point
estimates therefore states as fact things the deposition cannot constrain, and
the EMD4 build has already shown what that costs -- its uncertainty layer
overturned two of its own headline claims.

So the free rate constants receive independent log-normal stress-test
perturbations and every qualitative claim is re-scored across the draws. A claim
counts only if it survives the spread. These draws are NOT a fitted joint
distribution, a posterior, a confidence interval, or a credible interval: they
are a heuristic sensitivity band over parameters that were chosen, not measured.

What is NOT sampled
-------------------
The structural hypothesis is not sampled, because it is the model's claim rather
than its uncertainty:

- the segment profiles (`SEG_MICROBES`, `SEG_OXYGEN`, `SEG_KTRANSIT`,
  `SEG_ABSORB`, `SEG_VOL`) -- the shape of the gut;
- the provenance tagging itself;
- the arm capacities `gus` / `ox`, which define what an arm IS. The conventional
  arm is the unit of reference and the germ-free arm is structurally zero;
  neither is a free quantity. The one arm capacity that IS an assumption rather
  than a definition -- the two-member consortium's beta-glucuronidase -- gets its
  own dedicated sweep in :func:`two_member_gus_sensitivity`, because a
  log-normal centred on the nominal would explore it in the wrong direction: a
  two-species synthetic community cannot carry MORE beta-glucuronidase than the
  conventional community it was drawn from, so the honest question is how far
  DOWN the claim survives, not how far either way.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .model import Arm, Params, lumen_plasma_ratio, simulate, trace

# Free parameters and their log-scale sigma, graded by how well grounded each is.
FREE_SIGMA = {
    # Standard mouse physiology from PBPK references -- known to within tens of
    # per cent, not orders of magnitude.
    "V_plasma": 0.15, "V_bladder": 0.15, "V_kidney": 0.15,
    "dose_rate": 0.15, "Q_bile": 0.25,
    # Host disposition: plausible first-order constants, none measured here.
    "ka_BBN": 0.25, "k_uptake": 0.25, "k_gluc": 0.25, "k_bile": 0.25,
    "k_renal": 0.25, "k_kidney_out": 0.25, "k_void": 0.25,
    "k_clear_p": 0.25, "k_clear_BBN": 0.25,
    # Host omega-oxidation is the null's favourite knob and the quantity the
    # germ-free arm pins hardest. Widen it rather than flatter the contrast.
    "k_ox_host": 0.35,
    # BCPN absorption and back-diffusion carry the chemical argument that makes
    # the lumen/plasma ratio informative at all (ionised acid vs neutral parent).
    # The DIRECTION is chemistry; the magnitude is not.
    "ka_BCPN": 0.30, "k_back": 0.30,
    # Microbial capacities are explicitly chosen values, not measured ones.
    # They get the widest band in the build.
    "Vmax_gus": 0.40, "Km_gus": 0.40, "Vmax_ox": 0.40, "Km_ox": 0.40,
    # Urothelial exposure per unit luminal BCPN is a proportionality with no
    # measurement behind it anywhere in this build.
    "k_uro": 0.35,
    # DELIBERATELY ABSENT: k_bile_BCPN. It is a structural switch, not a rate
    # with a nominal value to perturb -- a log-normal multiplier on a default of
    # zero explores nothing. Like the two-member arm's gus it is swept
    # explicitly instead, in `nulls.biliary_escape`, because the honest question
    # is how far UP it can go before V7 breaks, not how far either way.
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


# The claims that are testable draw-by-draw. Each maps a per-arm observable dict
# to a bool, exactly as the corresponding entry in the validation battery does,
# so the audit scores the SAME statement the battery asserts -- not a paraphrase.
ROBUSTNESS = {
    "V1": lambda e: e["B"]["c_BCPN_cecum"] > 50.0 * e["Germ-Free"]["c_BCPN_cecum"],
    "V2": lambda e: (e["BA"]["c_BCPN_urine"] < 0.95 * e["B"]["c_BCPN_urine"]
                     and e["BA"]["AUC_uro"] < 0.95 * e["B"]["AUC_uro"]),
    "V3": lambda e: (e["Germ-Free"]["AUC_uro"] < 0.95 * e["BA"]["AUC_uro"]
                     and e["Germ-Free"]["micro_fraction"] < 1e-6),
    "V4": lambda e: (e["Monocolonized"]["c_BCPN_cecum"]
                     > 50.0 * e["Germ-Free"]["c_BCPN_cecum"]
                     and e["Monocolonized"]["AUC_uro"] > 0.80 * e["B"]["AUC_uro"]),
    "V5": lambda e: (e["2Mem"]["micro_fraction"] < 1e-6
                     and e["3Mem"]["micro_fraction"] > 0.5),
    "V6": lambda e: e["2Mem"]["AUC_uro"] > 1.5 * e["Germ-Free"]["AUC_uro"],
    # V7's null clause needs no per-draw refit. The host-only null IS the
    # germ-free structure, and its lumen/plasma ratio is invariant to all three
    # parameters the null is allowed to rescale: k_ox_host, k_renal and
    # k_clear_p change the amplitude of the host BCPN pool, and both the luminal
    # and the plasma term are proportional to that amplitude, so the ratio
    # cancels exactly (verified to 1e-12; see tests). The urine clause is an
    # identity -- the null is refitted to match urine by construction.
    "V7": lambda e: (e["Germ-Free"]["ratio_cecum"] < 1.0
                     and e["B"]["ratio_cecum"] > 10.0),
    "V8": lambda e: (e["2Mem"]["AUC_uro"] > e["BA"]["AUC_uro"]
                     and e["2Mem"]["ratio_cecum"] < 1.0),
    # The claim V8 USED to make, kept so the ensemble reports why it was
    # restated rather than quietly dropping it.
    "V8-superseded": lambda e: e["2Mem"]["AUC_uro"] >= 0.95 * e["B"]["AUC_uro"],
    "V12": lambda e: (e["2Mem"]["AUC_uro"] > e["BA"]["AUC_uro"]
                      and e["2Mem"]["AUC_uro_micro"] < 1e-9
                      and e["BA"]["AUC_uro_micro"] > e["2Mem"]["AUC_uro_micro"]
                      and e["B"]["AUC_uro_micro"] > e["BA"]["AUC_uro_micro"]),
}

# V9 uses a subsample of null refits. The full V10 profile is not repeated
# per draw; V11 declares the build's imported/modelled endpoint scope.
STRUCTURAL_CHECKS = ("V10", "V11")
UNAUDITED_REASON = {
    "V10": "conditional synthetic profiles are not recomputed for each draw",
    "V11": "implementation scope; no corresponding functional endpoints imported",
    "V7b": "biliary-route sensitivity is swept separately; its zero baseline is not perturbed",
}


def run_ensemble(arms, t: np.ndarray, ratio_segment: str = "cecum",
                 n: int = DEFAULT_N, seed: int = DEFAULT_SEED,
                 base: Params | None = None) -> dict:
    """Propagate parameter uncertainty through every arm and re-score the claims.

    Returns per-arm observable bands and, for each robustness-testable check,
    the fraction of draws in which the claim holds.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng(seed)
    base = base or Params()
    keys = [a.key for a in arms]

    draws: list[dict] = []
    n_failed = 0
    for _ in range(n):
        q = sample_params(rng, base)
        try:
            draws.append({a.key: _observe(q, a, t, ratio_segment) for a in arms})
        except Exception:
            n_failed += 1

    if not draws:
        raise RuntimeError("no ensemble draw integrated successfully")

    # Ratio to the conventional arm, formed WITHIN each draw. Banding the
    # marginal AUC of each arm and dividing by a fixed denominator would
    # massively overstate the uncertainty of a ratio: both arms move together
    # under a shared parameter draw, so the marginal spreads largely cancel.
    # Two-member absolute AUC spans 16-128 across the draws while the
    # two-member/conventional RATIO spans 0.55-1.74.
    ref = keys[0] if "B" not in keys else "B"
    for d in draws:
        # NOT named `base`: that is the Params object above, and rebinding it to
        # a float here worked only because nothing used it afterwards.
        ref_auc = d[ref]["AUC_uro"]
        for k in keys:
            d[k]["AUC_rel_conventional"] = (d[k]["AUC_uro"] / ref_auc
                                            if ref_auc > 0 else float("nan"))

    obs = list(draws[0][keys[0]].keys())
    bands = {k: {o: dict(zip(("p5", "p50", "p95"), (
        float(v) for v in np.percentile([d[k][o] for d in draws], [5, 50, 95]))))
        for o in obs} for k in keys}

    audit = {}
    for cid, fn in ROBUSTNESS.items():
        hits = [bool(fn(d)) for d in draws]
        audit[cid] = float(np.mean(hits))

    # The headline contrast, reported as a band rather than a single number.
    auc_2mem_over_b = np.array([d["2Mem"]["AUC_uro"] / d["B"]["AUC_uro"]
                                for d in draws]) if "2Mem" in keys else np.array([])
    return {
        "n_requested": n, "n_ok": len(draws), "n_failed": n_failed,
        "seed": seed, "bands": bands, "audit": audit,
        "auc_2mem_over_b": dict(zip(("p5", "p50", "p95"), (
            float(v) for v in np.percentile(auc_2mem_over_b, [5, 50, 95]))))
        if len(auc_2mem_over_b) else None,
        "interpretation": (
            "Heuristic sensitivity bands over parameters that were chosen, not "
            "measured. NOT a posterior, confidence interval or credible "
            "interval. Structure and arm capacities are not sampled."),
    }


def null_cost_subsample(t: np.ndarray, n: int = 40, seed: int = DEFAULT_SEED + 1,
                        base: Params | None = None) -> dict:
    """V9 across a subsample of draws.

    V9 asks what fold-change in hepatic omega-oxidation a host-only model must
    assert to explain the conventional AND antibiotic arms at once. Unlike the
    other claims this one cannot be re-scored from the arm observables: it needs
    a null refit per draw, which is why it runs on a subsample rather than the
    full ensemble.
    """
    from .nulls import host_parameter_cost

    rng = np.random.default_rng(seed)
    base = base or Params()
    folds = []
    for _ in range(n):
        q = sample_params(rng, base)
        try:
            folds.append(float(host_parameter_cost(q, t)["k_ox_fold_required"]))
        except Exception:
            continue
    if not folds:
        raise RuntimeError("no null-cost draw completed")
    a = np.asarray(folds)
    return {"n_ok": len(folds), "n_requested": n, "seed": seed,
            "fold": dict(zip(("p5", "p50", "p95"),
                             (float(v) for v in np.percentile(a, [5, 50, 95])))),
            "frac_above_1p5": float(np.mean(a > 1.5))}


def capacity_decomposition(p: Params, t: np.ndarray,
                           ratio_segment: str = "cecum") -> dict:
    """Which microbial activity carries which claim: deconjugation or oxidation?

    The arms vary `gus` and `ox` together, so every systemic check is scored
    against a bundle. Crossing them separates the two, and the result is not the
    obvious one:

    * `gus` alone (the two-member arm) roughly TRIPLES urothelial exposure, by
      returning deconjugated parent for further HOST oxidation;
    * `ox` alone raises luminal BCPN by three orders of magnitude but raises
      systemic exposure far less, because luminal BCPN is a poorly absorbed acid
      and ~87% of it leaves in faeces;
    * adding `ox` on top of `gus` LOWERS total urothelial exposure, because
      luminal conversion diverts parent that would otherwise have been absorbed
      and oxidised by the host into a species that mostly exits in stool.

    So in this model microbial oxidation is a net SINK for systemic exposure,
    and every systemic claim in the battery (V2, V3, V6, V8, V12) is carried by
    beta-glucuronidase. Only V1 -- luminal BCPN -- requires the oxidase. That is
    a structural consequence worth stating plainly rather than leaving for a
    reader to derive, because the source paper's causal story runs through
    conversion, and here conversion does not carry the systemic endpoint.
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

    # Which capacity does V2 (antibiotic depletion lowers urine AND AUC) need?
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
            "Microbial oxidation raises luminal BCPN by ~3 orders of magnitude "
            "and LOWERS total urothelial exposure, because luminal BCPN is a "
            "poorly absorbed acid that mostly leaves in faeces. Every systemic "
            "check in the battery is carried by beta-glucuronidase; only V1 "
            "requires the oxidase. V2 fails outright if the oxidase alone is "
            "depleted."),
    }


def dose_only_sensitivity(p: Params, t: np.ndarray,
                          multipliers=(0.7, 1.0, 1.3),
                          ratio_segment: str = "cecum") -> dict:
    """Does dose uncertainty move the reported ratios? Not zero, and not much.

    The build's homogeneity property is exact but is often quoted too broadly.
    The model is invariant under a JOINT rescaling of `(dose_rate, Vmax_gus,
    Km_gus, Vmax_ox, Km_ox)`, because
    `lam*V * lam*c / (lam*K + lam*c) = lam * V*c/(K+c)` and every other term is
    linear. Uncertainty in the drinking volume does NOT do that: it moves the
    dose while the microbial kinetics stay where they were put, and the luminal
    system then sits at a different point on the Michaelis-Menten curve.

    So `+/-30% on the drinking volume moves no ratio` -- which earlier releases
    asserted, citing homogeneity -- is the wrong statement with the right
    reason attached to it. The effect is about 1% over that range, which is
    small enough to be worth reporting honestly rather than rounding to zero.
    Both quantities are computed here so the manuscript can state them apart.
    """
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
            "Joint rescaling of dose with the microbial Vmax/Km is EXACTLY "
            "invariant (machine precision). Moving the dose alone is not: over "
            "+/-30% the caecum/plasma ratio moves by about 1%. Dose-convention "
            "and drinking-volume uncertainty are the second case, not the "
            "first, and must not be waved away by citing homogeneity."),
    }


def two_member_gus_sensitivity(p: Params, t: np.ndarray,
                               grid=(0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00),
                               ratio_segment: str = "cecum") -> dict:
    """How far down can the two-member arm's beta-glucuronidase go?

    The build hands the two-member consortium `gus = 1.00` -- the full
    beta-glucuronidase capacity of a conventional community, from two species.
    That is an assumption, and both surviving forms of the two-member claim rest
    on it, so it gets swept explicitly rather than defended in prose.
    """
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
