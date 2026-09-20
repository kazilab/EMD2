"""Run the EMD2 simulation.

    python -m emd2_simulation.run [--no-identifiability] [--no-ensemble]
                                  [--n-ensemble 400] [--no-figure]

Exit status is non-zero if any validation check fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import identifiability as idf
from . import nulls as nl
from .conditions import ARMS, BY_KEY, CHECKS, RATIO_SEGMENT
from .ensemble import (DEFAULT_N, STRUCTURAL_CHECKS, UNAUDITED_REASON,
                       capacity_decomposition, dose_only_sensitivity,
                       null_cost_subsample, run_ensemble,
                       two_member_gus_sensitivity)
from .model import (SEGMENTS, Params, auc_is_urine_rescaled,
                    lumen_plasma_ratio, simulate, trace)

T_END = 720.0                  # hours; ~30 days, well past steady state
N_T = 1441


def run_arms(p: Params, t: np.ndarray) -> dict:
    out = {}
    for a in ARMS:
        y = simulate(p, a, t)
        tr = trace(y, p)
        tr["ratio_cecum"] = lumen_plasma_ratio(y, p, RATIO_SEGMENT)
        out[a.key] = tr
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--empirical-only", action="store_true",
                    help="reproduce the published-data analysis and main figures without the optional mechanistic model")
    ap.add_argument("--no-identifiability", action="store_true")
    ap.add_argument("--no-ensemble", action="store_true")
    ap.add_argument("--n-ensemble", type=int, default=DEFAULT_N)
    ap.add_argument("--n-null-audit", type=int, default=40,
                    help="draws for the V9 null-refit subsample; 0 to skip")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--outdir", default="figures/output")
    args = ap.parse_args()
    if args.n_ensemble <= 0:
        ap.error("--n-ensemble must be a positive integer")

    from .empirical import write_outputs
    empirical = write_outputs(Path(args.outdir), figure=not args.no_figure)
    cecum = next(r for r in empirical["consortium"] if r["Tissue"] == "CEC")
    print("EMD2: secondary analysis of published functional experiments")
    print(f"  Consortium caecal BCPN difference: {cecum['mean_difference']:.4f} source-table µM")
    print(f"  95% bootstrap CI: {cecum['difference_ci95']}; Holm P={cecum['p_holm']:.6g}")
    print("  Acute measurements are descriptive; no physiological rates fitted.")
    if args.empirical_only:
        return 0

    p = Params()
    t = np.linspace(0.0, T_END, N_T)
    bar = "=" * 78
    print(bar)
    print("EMD2: illustrative microbiome-mediated toxicokinetics (BBN -> BCPN)")
    print("Assigned kinetics; no calibrated concentration or source classifier.")
    print("KCC1-related metabolism; KCC6 home and KCC7/8/10 links conditional.")
    print("KCC2 requires genotoxicity evidence; no functional responses modelled.")
    print("Coverage is LC-MS acquisitions, not independent biological n.")
    print(bar)

    runs = run_arms(p, t)
    end = {k: {o: (float(v[-1]) if hasattr(v, "__len__") else float(v))
               for o, v in tr.items()} for k, tr in runs.items()}

    base = end["B"]["AUC_uro"]
    print(f"\n{'-' * 78}\nArms at steady state\n{'-' * 78}")
    print(f"  {'arm':<30}{'cecum':>9}{'plasma':>9}{'urine':>9}"
          f"{'uroAUC':>9}{'micro%':>8}{'cec/pl':>9}")
    for a in ARMS:
        e = end[a.key]
        tag = "" if a.reference else "  (not used in assignment)"
        print(f"  {a.label:<30}{e['c_BCPN_cecum']:>9.3f}{e['c_BCPN_plasma']:>9.4f}"
              f"{e['c_BCPN_urine']:>9.3f}{e['AUC_uro'] / base * 100:>8.0f}%"
              f"{e['micro_fraction'] * 100:>7.0f}%{e['ratio_cecum']:>9.2f}{tag}")

    print(f"\n  BCPN by segment, conventional arm, and the lumen/plasma ratio")
    print(f"  each segment would give. SETTLED AGAINST THE MODEL: Suppl. Tables")
    print(f"  S7/S13 give a caecal peak declining distally (6.69 / 4.80 / 2.51")
    print(f"  uM at 3 wk, same ordering at 7 wk); this model peaks at the colon.")
    print(f"    {'segment':<10}{'nmol/mL':>12}{'lumen/plasma':>14}")
    y_conv = simulate(p, BY_KEY["B"], t)          # integrate once, not per segment
    for s in SEGMENTS:
        rr = lumen_plasma_ratio(y_conv, p, s)
        print(f"    {s:<10}{end['B']['c_BCPN_' + s]:>12.0f}{rr:>14.2f}")
    print(f"    {'plasma':<10}{end['B']['c_BCPN_plasma']:>12.2f}")
    print(f"    {'kidney':<10}{end['B']['c_BCPN_kidney']:>12.2f}")

    print(f"\n  Conventional vs antibiotic by compartment. Plasma, kidney and")
    print(f"  urine are serially connected here, so their contrasts are")
    print(f"  necessarily identical. The 'flat systemic BCPN' reading is")
    print(f"  WITHDRAWN: Suppl. Table S6 gives medians 2.55 / 2.81 / 3.75x in")
    print(f"  plasma/kidney/liver against this model's 2.11x, but its two")
    print(f"  analytical batches disagree in DIRECTION, so the systemic contrast")
    print(f"  is not resolvable in those data.")
    print(f"    {'compartment':<14}{'conv':>12}{'abx':>12}{'ratio':>9}")
    for lab, key in (("caecum", "c_BCPN_cecum"), ("plasma", "c_BCPN_plasma"),
                     ("kidney", "c_BCPN_kidney"), ("urine", "c_BCPN_urine")):
        b_, a_ = end["B"][key], end["BA"][key]
        print(f"    {lab:<14}{b_:>12.2f}{a_:>12.2f}{b_ / a_:>8.2f}x")

    # --- what the urothelial exposure index actually is ---------------------
    k = auc_is_urine_rescaled(p)
    obs = end["B"]["AUC_uro"] / end["B"]["U_total"]
    print(f"\n  The urothelial column is an ASSUMED EXPOSURE INDEX, not a tissue")
    print(f"  measurement, and it is not independent of urine. There is no")
    print(f"  urothelial compartment: under these equations")
    print(f"    AUC_uro / U_total = k_uro/(V_bladder*k_void) = {k:.10f}")
    print(f"    observed in the conventional arm                = {obs:.10f}")
    print(f"  exactly, in every arm. The index therefore supplies no information")
    print(f"  beyond cumulative urinary excretion, and no host-response evidence.")

    # --- does dose uncertainty move the ratios? -----------------------------
    dose = dose_only_sensitivity(p, t, ratio_segment=RATIO_SEGMENT)
    print(f"\n  Dose sensitivity, stated apart from the homogeneity property.")
    print(f"  Joint rescaling of (dose, Vmax, Km) is EXACTLY invariant; moving")
    print(f"  the dose ALONE is not, because the luminal system then sits at a")
    print(f"  different point on the Michaelis-Menten curve.")
    print(f"    {'dose x':>8}{'nmol/h':>10}{'cec/pl (dose only)':>21}"
          f"{'cec/pl (joint)':>17}")
    for d, j in zip(dose["dose_only"], dose["joint_rescaling"]):
        print(f"    {d['multiplier']:>8.2f}{d['dose_rate']:>10.1f}"
              f"{d['ratio_cecum']:>21.3f}{j['ratio_cecum']:>17.6f}")
    print(f"    max change over the range: dose only "
          f"{dose['max_rel_change_dose_only'] * 100:.2f}%, "
          f"joint {dose['max_rel_change_joint'] * 100:.2e}%")

    # --- which microbial activity carries which claim ----------------------
    decomp = capacity_decomposition(p, t, RATIO_SEGMENT)
    print(f"\n{'-' * 78}\nDeconjugation vs conversion: crossing the two capacities"
          f"\n{'-' * 78}")
    print("  The arms move `gus` and `ox` together, so every systemic check "
          "scores a\n  bundle. Crossed separately the result is not the obvious "
          "one.\n")
    print(f"  {'gus':>5}{'ox':>5}{'caecal BCPN':>14}{'plasma':>9}{'uroAUC':>10}"
          f"{'AUC host':>10}{'AUC micro':>11}{'cec/pl':>9}")
    for cell in decomp["cells"].values():
        print(f"  {cell['gus']:>5.2f}{cell['ox']:>5.2f}"
              f"{cell['c_BCPN_cecum']:>14.2f}{cell['c_BCPN_plasma']:>9.2f}"
              f"{cell['AUC_uro']:>10.0f}{cell['AUC_uro_host']:>10.0f}"
              f"{cell['AUC_uro_micro']:>11.0f}{cell['ratio_cecum']:>9.2f}")
    print(f"\n  Adding the oxidase to a deconjugating community changes total")
    print(f"  urothelial exposure by a factor of "
          f"{decomp['oxidase_effect_on_total_AUC']:.2f} -- it LOWERS it. "
          f"Microbial")
    print(f"  conversion is a net SINK for systemic exposure, even though it is")
    print(f"  73% of the conventional arm's exposure BY PROVENANCE.")
    print(f"  Luminal BCPN is a poorly absorbed acid")
    print(f"  (~87% leaves in faeces), so conversion diverts parent that would")
    print(f"  otherwise have been absorbed and oxidised by the host.")
    print(f"\n  Which capacity does V2 (antibiotic depletion) actually need?")
    print(f"    {'perturbation':<20}{'urine':>10}{'uroAUC':>10}   V2 holds?")
    for lab, v in decomp["v2_attribution"].items():
        print(f"    {lab:<20}{v['c_BCPN_urine']:>10.2f}{v['AUC_uro']:>10.0f}"
              f"   {'yes' if v['v2_holds'] else 'NO'}")
    print(f"    Every systemic check (V2, V3, V6, V8, V12) is carried by")
    print(f"    beta-glucuronidase. Only V1 -- luminal BCPN -- needs the oxidase.")

    # --- null comparison --------------------------------------------------
    print(f"\n{'-' * 78}\nModel comparison: can a host-only model do this?\n{'-' * 78}")
    null = nl.evaluate_null(p, t, RATIO_SEGMENT)
    null["host_cost"] = nl.host_parameter_cost(p, t)
    null["biliary_escape"] = nl.biliary_escape(p, t, RATIO_SEGMENT)
    print(f"  host-only null, refitted on {', '.join(nl.FREE)}")
    print("    rescaling applied: " + ", ".join(
        f"{k} x{v:.2f}" for k, v in null["scaling"].items()))
    print(f"    urinary BCPN   target {null['urine_target']:.2f}   "
          f"null {null['urine_null']:.2f}   relative error "
          f"{null['urine_rel_err']:.2e}")
    print("      (an exact recovery of ONE scalar by THREE free parameters --")
    print("       not a goodness-of-fit, and reported as such on purpose)")
    print(f"    caecal BCPN    mechanistic {null['cecum_mech']:.3f}   "
          f"null {null['cecum_null']:.4f}")
    print(f"    lumen/plasma   mechanistic {end['B']['ratio_cecum']:.2f}   "
          f"null {null['ratio']:.3f}")
    print(f"\n    The null recovers the urinary target exactly and still misses "
          f"the caecal\n    concentration by "
          f"{null['cecum_mech'] / max(null['cecum_null'], 1e-9):.0f}-fold.")
    hc = null["host_cost"]
    print(f"\n  What the null must assert to explain BOTH arms:")
    print(f"    hepatic omega-oxidation differs between conventional and "
          f"antibiotic\n    animals by {hc['k_ox_fold_required']:.2f}-fold.")
    print(f"    The microbial model requires a host change of "
          f"{hc['microbial_host_change']:.2f}-fold -- none.")
    print(f"    Quote the magnitude, not 'no mechanism': antibiotics are not")
    print(f"    inert toward the liver, and the one-knob fit is a well-posedness")
    print(f"    choice, not a claim that the change must fall on this parameter.")

    # --- V7b: the price of the headline's escape hatch ----------------------
    be = null["biliary_escape"]
    print(f"\n  V7b -- what the null would need to BEAT the headline statistic.")
    print("  The baseline contrast is conditional on assigned exchange and bile.")
    print("  Exchange reciprocity is not enforced; exchange changes can also")
    print("  raise a host-only ratio above one. Here we vary biliary delivery.")
    print(f"    {'k_bile_BCPN':>12}{'% refit clr':>13}{'% nominal':>11}"
          f"{'urine rel err':>15}{'null cec/pl':>13}   inverts?")
    for r in be["rows"]:
        print(f"    {r['k_bile_BCPN']:>12.4f}"
              f"{r['frac_of_plasma_clearance'] * 100:>12.1f}%"
              f"{r['frac_of_nominal_clearance'] * 100:>10.1f}%"
              f"{r['urine_rel_err']:>15.1e}{r['null_ratio']:>13.3f}"
              f"   {'YES -- V7 would break' if r['inverts_the_statistic'] else 'no'}")
    if be["min_frac_breaking_V7"] is not None:
        print(f"    Cheapest break on this grid: "
              f"{be['min_frac_breaking_V7'] * 100:.1f}% of the refitted null's "
              f"BCPN clearance\n    ({be['min_frac_nominal_breaking_V7'] * 100:.1f}% "
              f"of nominal), with urine still recovered exactly.")
        print("    V7 applies only to the tested baseline transport assumptions.")
    print("    The reported germ-free table ratio is a provisional comparison:")
    print("    matrix-specific normalisation is unresolved, so no biliary rate")
    print("    is empirically estimated by this sensitivity calculation.")

    # --- identifiability ---------------------------------------------------
    ident = None
    design = idf.load_design()
    scope = idf.host_response_verdict(design)
    if not args.no_identifiability:
        print(f"\n{'-' * 78}\nConditional synthetic recoverability\n{'-' * 78}")
        ident = idf.run(p, t)
        f = ident["facts"]
        print(f"  arms with urine sampled : {', '.join(f['arms_with_urine'])}")
        print(f"  arms with bile sampled  : {', '.join(f['arms_with_bile'])}")
        print(f"  disjoint?                 {f['urine_bile_disjoint']}")
        print("\n  Coverage counts LC-MS acquisitions; biological n is not inferred.")
        print(f"    {'compartment':<12}{'records':>9}{'arms':>6}")
        for pt, c in sorted(f["coverage"].items(), key=lambda kv: -kv[1]["records"]):
            print(f"    {pt:<12}{c['records']:>9}{c['n_arms']:>6}")
        print(f"    Anatomy/filename conflicts flagged: {len(f['metadata_conflicts'])}")
        print("    Synthetic terminal observations; fixed nuisance kinetics.")
        print("    Actual sample size, timing and covariance are not in the objective.")
        print("\n  Admissible Vmax_ox at 15% relative error. The ridge is scored on")
        print("  the WORST residual in the set, not the mean: a set that misses one")
        print("  compartment badly has identified nothing, and averaging hides it.")
        print(f"\n  {'measurement set':<32}{'admissible':<18}{'ridge':>7}"
              f"{'mean-metric':>13}   note")
        for key in idf.SETS:
            d = ident[key]
            adm = d["admissible"]
            rng = f"{adm.min():.2f} - {adm.max():.2f}" if len(adm) else "empty"
            notes = []
            if d["saturated"]:
                notes.append("UNBOUNDED on the tested grid")
            if d["uninformative_arms"]:
                notes.append("no information from "
                             + ",".join(d["uninformative_arms"]))
            if not d["contiguous"]:
                # Silent otherwise: max-min would report the gap as admissible.
                notes.append("NON-CONTIGUOUS -- width overstates the ridge")
            print(f"  {idf.SET_LABELS[key]:<32}{rng:<18}"
                  f"{d['ridge_width'] * 100:>6.0f}%"
                  f"{d['ridge_width_mean_metric'] * 100:>12.0f}%   "
                  f"{'; '.join(notes)}")

    # --- scope --------------------------------------------------------------
    print(f"\n{'-' * 78}\nScope: every functional downstream link\n{'-' * 78}")
    print(f"  deposition compartments: {', '.join(scope['compartments'])}")
    print(f"  imported endpoint types: {', '.join(scope['imported_endpoints'])}")
    print(f"  unmodelled functional edges: {', '.join(scope['unconstrainable_edges'])}")
    print("  KCC10 requires proliferation, cell-death or nutrient-supply evidence.")
    print(f"  basis: {scope['basis']}")

    # --- ensemble -----------------------------------------------------------
    ens = None
    gus = None
    if not args.no_ensemble:
        print(f"\n{'-' * 78}")
        print(f"Heuristic sensitivity propagation "
              f"({args.n_ensemble} independent log-normal draws)")
        print(f"{'-' * 78}")
        ens = run_ensemble(ARMS, t, RATIO_SEGMENT, n=args.n_ensemble)
        print(f"  {ens['n_ok']}/{ens['n_requested']} draws integrated successfully")
        print("  Bands are sensitivity ranges over CHOSEN parameters. They are not")
        print("  posteriors, confidence intervals or credible intervals.\n")
        print(f"  {'arm':<16}{'urothelial AUC (p5 / p50 / p95)':>40}"
              f"{'caecum/plasma p50':>20}")
        for a in ARMS:
            b = ens["bands"][a.key]
            z, r = b["AUC_uro"], b["ratio_cecum"]
            print(f"  {a.label[:15]:<16}"
                  f"{z['p5']:>12.2f}{z['p50']:>14.2f}{z['p95']:>14.2f}"
                  f"{r['p50']:>20.2f}")
        q = ens["auc_2mem_over_b"]
        print(f"\n  two-member / conventional urothelial AUC: "
              f"{q['p50']:.2f}  [{q['p5']:.2f}, {q['p95']:.2f}]")
        print("\n  Claim robustness across the draws:")
        for cid, frac in ens["audit"].items():
            mark = "robust" if frac >= 0.95 else ("marginal" if frac >= 0.80
                                                  else "NOT ROBUST")
            note = ("  <- superseded form, kept to show why V8 was restated"
                    if cid == "V8-superseded" else "")
            # COUNTS, not percentages. 400 judgement-based draws support three
            # significant figures nowhere: "99.8%" reads as a calibrated
            # probability when it is 399 successes out of 400.
            print(f"    {cid:<16}{round(frac * ens['n_ok']):>4d}/{ens['n_ok']:<6d}"
                  f"{mark}{note}")

        if args.n_null_audit > 0:
            nc = null_cost_subsample(t, n=args.n_null_audit)
            ens["null_cost"] = nc
            print(f"\n  V9 needs a null refit per draw, so it is audited on a "
                  f"subsample of {nc['n_ok']}:")
            print(f"    hepatic omega-oxidation fold-change the null must assert: "
                  f"{nc['fold']['p50']:.2f}  [{nc['fold']['p5']:.2f}, "
                  f"{nc['fold']['p95']:.2f}]")
            print(f"    exceeds 1.5-fold in "
                  f"{round(nc['frac_above_1p5'] * nc['n_ok'])} of {nc['n_ok']} draws")
        print(f"\n  V10 and V11 concern synthetic recovery and model scope, not the")
        print(f"  parameters: they do not vary across draws and are not audited.")

        gus = two_member_gus_sensitivity(p, t)
        print("\n  Two-member beta-glucuronidase sweep. The build hands this arm")
        print("  gus = 1.00 -- full conventional capacity from two species -- and")
        print("  both surviving forms of the claim depend on how far that can fall:")
        print(f"    {'gus':>6}{'uroAUC':>10}{'% conv':>9}   > abx arm?")
        for r in gus["rows"]:
            print(f"    {r['gus']:>6.2f}{r['AUC_uro']:>10.2f}"
                  f"{r['frac_of_conventional'] * 100:>8.0f}%   {r['exceeds_abx']}")
        print(f"    lowest gus still beating the antibiotic arm: "
              f"{gus['min_gus_beating_abx']}")

    # --- validation --------------------------------------------------------
    ctx = {"end": end, "null": null, "scope": scope,
           "ident": ident or {"urine_only_degenerate": None, "skipped": True}}
    print(f"\n{bar}\nValidation battery\n{bar}")
    results = []
    audit = ens["audit"] if ens else {}
    for chk in CHECKS:
        if chk.cid == "V10" and ctx["ident"].get("skipped"):
            print(f"\n[SKIP] {chk.cid}  {chk.edge}")
            print(f"       {chk.statement}")
            print("       basis: skipped (--no-identifiability); not counted")
            continue
        try:
            ok = bool(chk.test(ctx))
        except Exception as exc:
            ok = False
            print(f"  !! {chk.cid} raised: {exc}")
        results.append((chk, ok))
        print(f"\n[{'PASS' if ok else 'FAIL'}] {chk.cid}  {chk.edge}")
        print(f"       {chk.statement}")
        if chk.cid in audit:
            print(f"       robustness: holds in "
                  f"{round(audit[chk.cid] * ens['n_ok'])} of {ens['n_ok']} "
                  f"ensemble draws")
        elif ens and chk.cid == "V9" and "null_cost" in ens:
            nc = ens["null_cost"]
            print(f"       robustness: holds in "
                  f"{round(nc['frac_above_1p5'] * nc['n_ok'])} of {nc['n_ok']} "
                  f"draws (null refit per draw)")
        elif ens and chk.cid in UNAUDITED_REASON:
            print(f"       robustness: {UNAUDITED_REASON[chk.cid]}")
        print(f"       basis: {chk.source}")
    n_pass = sum(1 for _, o in results if o)
    print(f"\n  {n_pass}/{len(results)} checks passed")

    # --- summary -----------------------------------------------------------
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "empirical": {"file": "emd2_empirical_summary.json", "input_sha256": empirical["input_sha256"],
                      "analysis_kind": empirical["analysis_kind"], "kinetic_calibration": empirical["kinetic_calibration"]},
        "arms": {k: {o: v for o, v in e.items()} for k, e in end.items()},
        "null": {kk: vv for kk, vv in null.items()
                 if kk not in ("params", "host_cost")},
        "null_host_cost": {"k_ox_fold_required":
                           null["host_cost"]["k_ox_fold_required"]},
        "capacity_decomposition": decomp,
        "dose_sensitivity": dose,
        "exposure_index": {
            "auc_over_urine": auc_is_urine_rescaled(p),
            "note": ("AUC_uro is k_uro/(V_bladder*k_void) times cumulative "
                     "urinary excretion, exactly, in every arm. It is an "
                     "assumed exposure index with no urothelial compartment "
                     "behind it and carries no information beyond U_total."),
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
                           # max-min is a ridge WIDTH only if the admissible set
                           # is one interval; a split set would report its gap
                           # as admissible. Carried so the claim is checkable.
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
    path = outdir / "emd2_simulation_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n  wrote {path}")

    if not args.no_figure:
        try:
            from .figure import make_figure
            make_figure(runs, t, ctx, ident, outdir, ens)
        except ImportError:
            print("  (figure module not present yet; skipped)")

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
