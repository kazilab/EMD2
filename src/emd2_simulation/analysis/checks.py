"""Structural checks for the assigned-kinetics model."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Check:
    cid: str
    edge: str
    statement: str
    test: object
    source: str


def _gt(a, b, margin=1.05):
    return a > b * margin


def _lt(a, b, margin=0.95):
    return a < b * margin


CHECKS = [
    Check('V1', 'KCC1-related model contrast',
          "At the baseline parameters, conventional caecal BCPN exceeds the germ-free "
          "simulation by more than 50-fold.",
          lambda c: _gt(c["end"]["B"]["c_BCPN_cecum"],
                        c["end"]["Germ-Free"]["c_BCPN_cecum"], margin=50.0),
          "Host kinetics are held fixed in this comparison."),

    Check('V2', 'assigned microbial depletion',
          "Reducing assigned microbial capacities lowers simulated urinary BCPN and the"
          " exposure index with host kinetics fixed.",
          lambda c: (_lt(c["end"]["BA"]["c_BCPN_urine"], c["end"]["B"]["c_BCPN_urine"])
                     and _lt(c["end"]["BA"]["AUC_uro"], c["end"]["B"]["AUC_uro"])),
          "In the reference model, deconjugation accounts for the systemic decrease "
          "under microbial depletion. The study attributes the effect to conversion."),

    Check('V3', 'assigned germ-free scenario',
          "The germ-free simulation has lower exposure than the antibiotic scenario and"
          " zero microbial-origin BCPN.",
          lambda c: (_lt(c["end"]["Germ-Free"]["AUC_uro"], c["end"]["BA"]["AUC_uro"])
                     and c["end"]["Germ-Free"]["micro_fraction"] < 1e-6),
          "The germ-free arm sets both microbial capacities to zero and assumes the same host kinetics."),

    Check('V4', 'assigned mono-colonisation',
          "The assigned converting isolate restores simulated luminal BCPN and exposure"
          " toward conventional.",
          lambda c: (_gt(c["end"]["Monocolonized"]["c_BCPN_cecum"],
                         c["end"]["Germ-Free"]["c_BCPN_cecum"], margin=50.0)
                     and c["end"]["Monocolonized"]["AUC_uro"]
                     > 0.80 * c["end"]["B"]["AUC_uro"]),
          "Roje et al. motivate this scenario. Saturation limits the sensitivity of "
          "the prediction to the assigned microbial capacity."),

    Check('V5', 'assigned consortium reconstruction',
          "The two-member community has zero assigned oxidation. Adding an "
          "oxidising isolate restores microbial-origin BCPN in the model.",
          lambda c: (c["end"]["2Mem"]["micro_fraction"] < 1e-6
                     and c["end"]["3Mem"]["micro_fraction"] > 0.5),
          "Roje et al. motivate these consortium arms. Their model capacities are assigned."),

    Check('V6', 'enterohepatic recycling prediction',
          "Deconjugation alone increases simulated host-derived exposure relative to "
          "the germ-free scenario.",
          lambda c: _gt(c["end"]["2Mem"]["AUC_uro"],
                        c["end"]["Germ-Free"]["AUC_uro"], margin=1.5),
          "Host metabolism can produce BCPN in the absence of microbial oxidation. "
          "The exposure magnitude lacks calibration against animal data."),

    Check('V7', 'baseline host-only comparison',
          "At the tested parameters, a refitted host-only model matches simulated urine"
          " while its caecum/plasma ratio remains below one.",
          lambda c: (c["null"]["urine_rel_err"] < 1e-6
                     and c["null"]["ratio"] < 1.0
                     and c["end"]["B"]["ratio_cecum"] > 10.0),
          "The target is simulated conventional urine. Independent exchange coefficients "
          "can yield host-only ratios above one even without biliary BCPN delivery."),

    Check('V7b', 'biliary-route sensitivity',
          "Adding biliary BCPN delivery allows a refitted host-only model to exceed a "
          "ratio of one while matching simulated urine.",
          lambda c: (c["null"]["biliary_escape"]["min_frac_breaking_V7"]
                     is not None),
          "The sweep assumes biliary delivery and expresses it as a fraction of "
          "plasma clearance in the refitted host-only model."),

    Check('V8', 'exposure versus assigned conversion',
          "In the two-member simulation, the exposure index exceeds that in the "
          "antibiotic arm and the caecum/plasma ratio is below one.",
          lambda c: (c["end"]["2Mem"]["AUC_uro"] > c["end"]["BA"]["AUC_uro"]
                     and c["end"]["2Mem"]["ratio_cecum"] < 1.0),
          "The two-member arm has zero assigned microbial oxidation. Other mechanisms can yield a low ratio."),

    Check('V9', 'host-parameter sensitivity',
          "A one-parameter host-only refit requires a greater than 1.5-fold hepatic-"
          "oxidation contrast to reproduce the two simulated urinary targets.",
          lambda c: c["null"]["host_cost"]["k_ox_fold_required"] > 1.5,
          "The 1.5-fold threshold is heuristic. Holding the remaining host parameters "
          "fixed assigns the entire contrast to hepatic oxidation."),

    Check('V10', 'conditional synthetic recoverability',
          "The admissible range for urine-only synthetic data reaches a tested grid boundary.",
          lambda c: c["ident"]["urine_only_degenerate"],
          "The profile uses synthetic terminal values, one refitted nuisance "
          "parameter and a 15% residual threshold. Experimental identifiability "
          "and statistical power remain untested."),

    Check('V11', 'functional response scope',
          "The model lacks functional KCC6, KCC7, KCC8, KCC2 and KCC10 endpoints.",
          lambda c: (c["scope"]["host_response_unidentifiable"]
                     and c["scope"]["genotoxicity_unidentifiable"]
                     and c["scope"]["kcc10_response_unidentifiable"]),
          "KCC6 requires chronic-inflammatory evidence, KCC7 immunosuppression, KCC8 "
          "receptor-mediated effects, KCC2 genotoxicity, and KCC10 altered "
          "proliferation, cell death or nutrient supply (Smith et al., 2016)."),

    Check('V12', 'total and source-resolved exposure',
          "Total and microbial-origin exposure indices rank the two-member and "
          "antibiotic arms differently.",
          lambda c: (c["end"]["2Mem"]["AUC_uro"] > c["end"]["BA"]["AUC_uro"]
                     and c["end"]["2Mem"]["AUC_uro_micro"] < 1e-9
                     and c["end"]["BA"]["AUC_uro_micro"]
                     > c["end"]["2Mem"]["AUC_uro_micro"]
                     and c["end"]["B"]["AUC_uro_micro"]
                     > c["end"]["BA"]["AUC_uro_micro"]),
          "BCPN from either source contributes to total exposure. The model has no "
          "dose-response relationship linking exposure to genotoxicity or tumour risk."),

]


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
    # The host-only lumen/plasma ratio is invariant to the three refitted
    # host rates, so the germ-free ratio evaluates the contrast in each draw.
    "V7": lambda e: (e["Germ-Free"]["ratio_cecum"] < 1.0
                     and e["B"]["ratio_cecum"] > 10.0),
    "V8": lambda e: (e["2Mem"]["AUC_uro"] > e["BA"]["AUC_uro"]
                     and e["2Mem"]["ratio_cecum"] < 1.0),
    "two_member_auc_at_least_95pct_conventional": lambda e: e["2Mem"]["AUC_uro"] >= 0.95 * e["B"]["AUC_uro"],
    "V12": lambda e: (e["2Mem"]["AUC_uro"] > e["BA"]["AUC_uro"]
                      and e["2Mem"]["AUC_uro_micro"] < 1e-9
                      and e["BA"]["AUC_uro_micro"] > e["2Mem"]["AUC_uro_micro"]
                      and e["B"]["AUC_uro_micro"] > e["BA"]["AUC_uro_micro"]),
}

# V9 uses its own sample of host-only refits.
STRUCTURAL_CHECKS = ("V10", "V11")
UNAUDITED_REASON = {
    "V10": "synthetic profiles use the reference parameters",
    "V11": "functional endpoints are absent from the model",
    "V7b": "a separate sensitivity grid varies biliary BCPN delivery",
}

