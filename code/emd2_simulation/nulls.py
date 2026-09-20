"""Host-only comparisons fitted to synthetic conventional urinary BCPN.

The null removes microbial reactions and refits three host parameters to one
simulated scalar. Its exact urinary recovery demonstrates non-uniqueness, not
empirical goodness-of-fit. The baseline caecum/plasma contrast is conditional
on assigned transport. Independent exchange coefficients impose no physical
reciprocity bound; both exchange changes and biliary delivery can raise the
host-only ratio above one. Biliary rates below are stress-test assumptions.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from .model import Arm, Params, lumen_plasma_ratio, simulate, trace

# The null is germ-free by construction: no deconjugation, no microbial oxidation.
HOST_ONLY = Arm("null", "Host-only null", gus=0.0, ox=0.0)

# Host parameters the null may rescale. Deliberately generous -- three degrees of
# freedom covering generation, renal handling and systemic loss, which is more
# than enough to hit a single scalar urinary target and is chosen so that
# failure cannot be blamed on a handicapped alternative.
FREE = ["k_ox_host", "k_renal", "k_clear_p"]


def fit_host_only(p: Params, target_urine: float,
                  t: np.ndarray) -> tuple[Params, float]:
    """Rescale host parameters so the host-only model matches urinary BCPN.

    Returns the refitted parameters and the RELATIVE ERROR against the target.
    Not an R^2: three free parameters are fitted to one scalar observation, so
    the residual has zero degrees of freedom and the variance-explained ratio is
    undefined -- it evaluated to exactly 1.0 by construction, which said nothing
    about the null and read as a perfect fit.
    """
    base = np.array([getattr(p, k) for k in FREE])

    def resid(logmul):
        q = replace(p, **{k: float(b * np.exp(m))
                          for k, b, m in zip(FREE, base, logmul)})
        try:
            tr = trace(simulate(q, HOST_ONLY, t), q)
        except Exception:
            # Scaled to the target, not absolute: a fixed 1e3 sentinel is
            # SMALLER than a typical residual once the dose is on a real mass
            # balance, so a failed integration would have scored as a good fit.
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
    """Fit the host-only null to the conventional arm's urine, then confront it
    with the compartment-resolved simulations it was not fitted to."""
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
        # Relative residual only. The null is refitted to hit ONE number with
        # three free parameters, so what matters is that it recovers the target
        # to optimiser tolerance, not how well it "fits" -- there is no spread
        # to explain and no goodness-of-fit statistic to quote.
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

    Clearance fractions use each refitted null's plasma clearance. All grid
    values are sensitivity settings, not estimates fitted to animal data.
    The legacy intermediate grid point 0.0227/h is retained for reproducibility.
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
        # What fraction of BCPN leaving plasma goes to bile. Quoted against the
        # REFITTED null's own clearance, which is the honest denominator -- the
        # null rescales k_renal and k_clear_p to hit urine, so its clearance is
        # not the nominal one. The nominal figure is carried alongside because
        # it is the larger of the two and should not be the only one reported.
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
            "The baseline contrast is conditional on assigned exchange and zero "
            "biliary delivery. No passive reciprocity constraint is enforced. "
            "Exchange changes or biliary delivery can raise a host-only ratio "
            "above one; this sweep refits simulated urine, not animal data."),
    }


def fit_host_only_single(p: Params, target_urine: float,
                         t: np.ndarray, knob: str = "k_ox_host") -> Params:
    """Match a urinary target by moving ONE host parameter.

    ``fit_host_only`` has three free parameters against a single scalar target,
    so it is underdetermined: it can hit any urine value by adjusting renal
    handling and leave hepatic oxidation essentially untouched. That is fine for
    asking "can the null match urine at all", and useless for asking "by how much
    must hepatic oxidation differ", which is only well posed with one knob.
    """
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
    """What must the host do, for a host-only model to explain BOTH arms?

    The manuscript sets this as an explicit criterion: microbial mediation is
    supported when the model "reproduces the effects of microbiota depletion or
    reconstruction WITHOUT implausible changes to unrelated host parameters."

    A host-only model can always match any single arm's urinary BCPN -- it has
    free hepatic and renal parameters. The question is what it must assert to
    match the conventional and the antibiotic arm at the same time. Since it has
    no microbial term, the entire difference has to be loaded onto host
    metabolism, which amounts to claiming that antibiotics changed hepatic
    omega-oxidation with no proposed mechanism.
    """
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
        # the microbial model needs NO host change at all: the arms differ only
        # in gus/ox, which are properties of the community
        "microbial_host_change": 1.0,
    }
