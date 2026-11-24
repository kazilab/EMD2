"""Summarise model endpoints, structural checks and sensitivity analyses."""
from __future__ import annotations
import numpy as np
from ..simulation.model import auc_is_urine_rescaled
from . import identifiability as idf
from .checks import CHECKS, STRUCTURAL_CHECKS, UNAUDITED_REASON


def analyze_simulation(result, empirical: dict | None = None) -> dict:
    """Summarise the configured model run and its completed diagnostics."""
    settings = result.settings
    if not np.array_equal(result.time, settings.time_grid()):
        raise ValueError("Simulation time grid no longer matches the recorded settings")
    p, t, end = result.params, result.time, result.endpoints
    null, dose, decomp = result.null, result.dose_sensitivity, result.capacity_decomposition
    scope, ident, ens, gus = result.scope, result.identifiability, result.ensemble, result.gus_sensitivity
    ctx = result.ctx
    results, register, skipped = [], {}, {}
    for check in CHECKS:
        if check.cid == "V10" and ident is None:
            skipped[check.cid] = "Synthetic profiles were disabled"
            register[check.cid] = {"edge": check.edge, "statement": check.statement,
                                   "source": check.source, "status": "not_run", "passed": None}
            continue
        try:
            passed = bool(check.test(ctx))
        except Exception as exc:
            raise RuntimeError(f"Internal check {check.cid} could not be evaluated") from exc
        results.append((check, passed))
        register[check.cid] = {"edge": check.edge, "statement": check.statement,
                               "source": check.source,
                               "status": "passed" if passed else "failed", "passed": passed}
    summary = {
        "arms": {k: {o: v for o, v in e.items()} for k, e in end.items()},
        "null": {kk: vv for kk, vv in null.items()
                 if kk not in ("params", "host_cost")},
        "null_host_cost": {"k_ox_fold_required":
                           null["host_cost"]["k_ox_fold_required"]},
        "capacity_decomposition": decomp,
        "dose_sensitivity": dose,
        "exposure_index": {
            "auc_over_urine": auc_is_urine_rescaled(p),
            "note": ("In every arm, AUC_uro equals U_total multiplied by "
                     "k_uro/(V_bladder*k_void). The index contains no independent "
                     "measurement of tissue exposure."),
        },
        "scope": scope,
        "checks": {c.cid: bool(o) for c, o in results},
    }
    if ident is not None:
        summary["identifiability"] = {
            "facts": ident["facts"],
            "metric": ("worst relative residual in the measurement set, "
                       "15% relative error"),
            "sets": {key: {"ridge_width": ident[key]["ridge_width"],
                           "ridge_width_mean_metric":
                               ident[key]["ridge_width_mean_metric"],
                           "saturated": ident[key]["saturated"],
                           "contiguous": ident[key]["contiguous"],
                           "uninformative_arms": ident[key]["uninformative_arms"]}
                     for key in idf.SETS},
        }
    if ens is not None:
        summary["ensemble"] = {
            "n_ok": ens["n_ok"], "n_requested": ens["n_requested"],
            "seed": ens["seed"],
            "bands": ens["bands"], "audit": ens["audit"],
            "auc_2mem_over_b": ens["auc_2mem_over_b"],
            "null_cost": ens.get("null_cost"),
            "structural_checks_not_audited": list(STRUCTURAL_CHECKS),
            "unaudited_reason": UNAUDITED_REASON,
            "interpretation": ens["interpretation"],
        }
        summary["two_member_gus_sensitivity"] = gus
    if empirical is not None:
        summary = {"empirical": {
            "file": "emd2_empirical_summary.json", "input_sha256": empirical["input_sha256"],
            "analysis_kind": empirical["analysis_kind"],
            "kinetic_calibration": empirical["kinetic_calibration"]}, **summary}
    summary["model_checks"] = register
    summary["diagnostics"] = {
        "identifiability_performed": ident is not None,
        "ensemble_performed": ens is not None,
        "null_cost_audit_performed": bool(ens is not None and ens.get("null_cost") is not None),
        "checks_total": len(CHECKS), "checks_evaluated": len(results),
        "checks_passed": sum(passed for _, passed in results),
        "skipped_checks": skipped,
        "complete": bool(ident is not None and ens is not None and ens.get("null_cost") is not None),
        "interpretation": "Pass counts cover evaluated computational checks. Biological validation requires independent evidence.",
    }
    return summary
