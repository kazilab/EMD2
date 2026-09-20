"""Arms and the validation battery.

Arm definitions follow the `Factor Value[Abx treatment]` levels in
`data/s_MTBLS3581.txt`, so the model and the deposition use the same names.

NO arm is fitted: the deposited MAF carries no quantitative values, so every
capacity below is STRUCTURALLY ASSIGNED rather than estimated. Reference arms
are the ones whose assignment defines the scale (conventional = 1, germ-free =
0) or is taken as given (antibiotics = 0.05).

A NOTE ON "HELD OUT", which this file used to overstate. Because nothing is
calibrated, there is no fit from which an arm could be withheld, and the
non-reference arms are not an independent validation set in the usual sense.
Their capacities are assigned by the same judgement as the reference arms'.
What is true, and weaker, is that no reference capacity was chosen to make a
non-reference arm come out right. `reference=False` marks that, and nothing
more. The consortium arms are still the most informative of them, because the
two-member community carries beta-glucuronidase but cannot oxidise, so `ox = 0`
is a structural zero rather than a point on a saturating curve.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Arm

ARMS = [
    Arm("B", "Conventional", gus=1.00, ox=1.00, plotted=True, reference=True),
    Arm("BA", "Antibiotics", gus=0.05, ox=0.05, plotted=True, reference=True),
    Arm("Germ-Free", "Germ-free", gus=0.00, ox=0.00, plotted=True, reference=True),
    # --- not used in assigning any reference capacity --------------------
    Arm("Monocolonized", "Mono-colonised", gus=0.70, ox=0.70),
    Arm("2Mem", "Two-member (GUS, no oxidase)", gus=1.00, ox=0.00, plotted=True),
    Arm("3Mem", "Three-member", gus=0.90, ox=0.75),
]
BY_KEY = {a.key: a for a in ARMS}
PLOTTED = [a for a in ARMS if a.plotted]
NON_REFERENCE = [a for a in ARMS if not a.reference]

# Caecum is the candidate ratio compartment. Coverage counts acquisitions,
# not independent animals; no biological n is inferred from filenames.
RATIO_SEGMENT = "cecum"


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
          "Motivated by the source study microbial transformation experiments. This "
          "comparison holds host parameters fixed and does not exclude all host-only "
          "alternatives."),

    Check('V2', 'assigned microbial depletion',
          "Reducing assigned microbial capacities lowers simulated urinary BCPN and the"
          " exposure index with host kinetics fixed.",
          lambda c: (_lt(c["end"]["BA"]["c_BCPN_urine"], c["end"]["B"]["c_BCPN_urine"])
                     and _lt(c["end"]["BA"]["AUC_uro"], c["end"]["B"]["AUC_uro"])),
          "The antibiotic scenario changes deconjugation and oxidation together. "
          "Capacity decomposition shows that deconjugation carries the systemic "
          "decrease in this model; microbial oxidation alone lowers total systemic "
          "exposure. This differs from the source study conversion mechanism."),

    Check('V3', 'assigned germ-free scenario',
          "The germ-free simulation has lower exposure than the antibiotic scenario and"
          " zero microbial-origin BCPN.",
          lambda c: (_lt(c["end"]["Germ-Free"]["AUC_uro"], c["end"]["BA"]["AUC_uro"])
                     and c["end"]["Germ-Free"]["micro_fraction"] < 1e-6),
          "The microbial capacities are set to zero; unchanged host kinetics are a "
          "modelling assumption, not a demonstrated effect of germ-free status."),

    Check('V4', 'assigned mono-colonisation',
          "The assigned converting isolate restores simulated luminal BCPN and exposure"
          " toward conventional.",
          lambda c: (_gt(c["end"]["Monocolonized"]["c_BCPN_cecum"],
                         c["end"]["Germ-Free"]["c_BCPN_cecum"], margin=50.0)
                     and c["end"]["Monocolonized"]["AUC_uro"]
                     > 0.80 * c["end"]["B"]["AUC_uro"]),
          "Motivated by mono-colonisation in Roje et al. Capacities are assigned, not "
          "estimated. Saturation makes the prediction insensitive over a broad capacity"
          " range; it is not independent quantitative validation."),

    Check('V5', 'assigned consortium reconstruction',
          "The assigned two-member community has zero microbial oxidation; adding an "
          "oxidising isolate restores microbial-origin BCPN.",
          lambda c: (c["end"]["2Mem"]["micro_fraction"] < 1e-6
                     and c["end"]["3Mem"]["micro_fraction"] > 0.5),
          "Motivated by the study isolate and reconstructed-community experiments. All "
          "capacities are assigned; these arms are not a held-out validation dataset."),

    Check('V6', 'enterohepatic recycling prediction',
          "Deconjugation alone increases simulated host-derived exposure relative to "
          "the germ-free scenario.",
          lambda c: _gt(c["end"]["2Mem"]["AUC_uro"],
                        c["end"]["Germ-Free"]["AUC_uro"], margin=1.5),
          "This is a conditional model prediction. Lack of microbial conversion does "
          "not imply lack of total BCPN. Its magnitude is not calibrated to animal "
          "data."),

    Check('V7', 'baseline host-only comparison',
          "At the tested parameters, a refitted host-only model matches simulated urine"
          " while its caecum/plasma ratio remains below one.",
          lambda c: (c["null"]["urine_rel_err"] < 1e-6
                     and c["null"]["ratio"] < 1.0
                     and c["end"]["B"]["ratio_cecum"] > 10.0),
          "The target is simulated conventional urine. k_back and ka_BCPN are "
          "independent and passive reciprocity is not enforced; k_back=0.05/h gives a "
          "host-only ratio above one even without bile. The check and ensemble "
          "demonstrate a contrast over tested parameters, not a physical bound or "
          "unique source attribution."),

    Check('V7b', 'biliary-route sensitivity',
          "Adding biliary BCPN delivery allows a refitted host-only model to exceed a "
          "ratio of one while matching simulated urine.",
          lambda c: (c["null"]["biliary_escape"]["min_frac_breaking_V7"]
                     is not None),
          "At k_bile_BCPN=0.05/h the ratio is approximately 1.37; the clearance "
          "fraction is calculated relative to the refitted host-only plasma clearance "
          "and reported in the results. This is a structural counterexample, not a "
          "measured biliary rate. Exchange changes provide another counterexample."),

    Check('V8', 'exposure versus assigned conversion',
          "The two-member simulation exceeds the antibiotic exposure index despite zero"
          " assigned microbial oxidation; its baseline ratio is below one.",
          lambda c: (c["end"]["2Mem"]["AUC_uro"] > c["end"]["BA"]["AUC_uro"]
                     and c["end"]["2Mem"]["ratio_cecum"] < 1.0),
          "This compares assigned simulations, not animal outcomes. A low ratio in this"
          " parameterisation does not diagnose zero microbial conversion in general."),

    Check('V9', 'host-parameter sensitivity',
          "A one-parameter host-only refit requires a greater than 1.5-fold hepatic-"
          "oxidation contrast to reproduce the two simulated urinary targets.",
          lambda c: c["null"]["host_cost"]["k_ox_fold_required"] > 1.5,
          "The 1.5-fold criterion is heuristic. Holding other parameters fixed assigns "
          "the contrast to hepatic oxidation; alternative host changes are possible. "
          "Report the ensemble interval without interpreting the cutoff as a biological"
          " plausibility boundary."),

    Check('V10', 'conditional synthetic recoverability',
          "With fixed nuisance kinetics, urine-only synthetic data admit the tested "
          "microbial-capacity range.",
          lambda c: c["ident"]["urine_only_degenerate"],
          "Measurement sets use recorded arm/compartment coverage. Terminal synthetic "
          "values, one fitted nuisance parameter and a 15% residual threshold do not "
          "establish empirical identifiability, statistical power or sufficiency of "
          "existing samples."),

    Check('V11', 'functional response scope',
          "No functional KCC6, KCC7, KCC8, KCC2 or KCC10 response is imported or "
          "modelled in this build.",
          lambda c: (c["scope"]["host_response_unidentifiable"]
                     and c["scope"]["genotoxicity_unidentifiable"]
                     and c["scope"]["kcc10_response_unidentifiable"]),
          "KCC6 requires chronic-inflammatory evidence, KCC7 immunosuppression, KCC8 "
          "receptor-mediated effects, KCC2 genotoxicity, and KCC10 altered "
          "proliferation, cell death or nutrient supply (Smith et al., 2016). Anatomy "
          "labels cannot establish assay absence from the deposition or source study."),

    Check('V12', 'total and source-resolved exposure',
          "Total and microbial-origin exposure indices rank assigned arms differently; "
          "neither is tested against genotoxicity, functional host responses or tumour "
          "incidence.",
          lambda c: (c["end"]["2Mem"]["AUC_uro"] > c["end"]["BA"]["AUC_uro"]
                     and c["end"]["2Mem"]["AUC_uro_micro"] < 1e-9
                     and c["end"]["BA"]["AUC_uro_micro"]
                     > c["end"]["2Mem"]["AUC_uro_micro"]
                     and c["end"]["B"]["AUC_uro_micro"]
                     > c["end"]["BA"]["AUC_uro_micro"]),
          "Provenance labels do not confer different chemical potency. Both pools "
          "contribute to total BCPN exposure; no dose-response or tumour-risk model is "
          "implemented. KCC2 and KCC10 require their corresponding functional "
          "endpoints, and tumour outcomes can provide additional evidence for the "
          "overall sequence."),

]
