"""EMD2: illustrative microbiome-mediated toxicokinetics of BBN -> BCPN.

The model compares candidate measurements under assigned kinetics. It is not
calibrated to animal concentrations and does not establish any KCC response.
Functional transformation and microbial perturbation evidence comes from the
source study (Roje et al., Nature 632, 2024), not from simulation output.

Host and microbial BCPN are chemically identical. Their provenance pools are
latent bookkeeping states; each ordinary concentration assay observes the sum.
At the baseline parameters a refitted host-only model matches simulated urine
but has a lower caecum/plasma ratio. This is a conditional model contrast, not
a unique source classifier. The independently assigned exchange coefficients
do not enforce passive reciprocity: k_back=0.05/h gives a host-only ratio above
one even with k_bile_BCPN=0. Biliary delivery also raises the ratio; its default
of zero is an unmeasured structural assumption (see nulls.biliary_escape).

Paired lumen/systemic measurements, preferably over time and accompanied by
functional microbial perturbation, can constrain alternatives when host
metabolism, transport and matrix-specific assay normalisation are considered.
The current endpoint data do not establish lumen-first or host-first timing.

The kidney is a serial elimination compartment. Under the synthetic profiling
assumptions, adding kidney to luminal measurements narrows the admissible range;
this does not establish empirical identifiability or experimental sample size.

Amounts, volumes and time are in nmol, mL and h. The illustrative drinking-water
input is 598 nmol/h, treating 0.05% as w/v. Microbial Vmax and Km are assigned.
Jointly rescaling dose, Vmax and Km preserves ratios, whereas dose-only changes
need not. Neither absolute concentrations nor ratios are validated predictions.
The cumulative exposure index is exactly proportional to urinary excretion.
KCC6 (conditional home), KCC7, KCC8 and KCC10 require corresponding functional
host-response evidence; KCC2 requires genotoxicity evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, fields

import numpy as np
from scipy.integrate import solve_ivp

# --- gut segments, proximal to distal -------------------------------------
# Named to match the `Organism part` levels in s_MTBLS3581.txt so the model and
# the deposition speak the same vocabulary.
SEGMENTS = ["duodenum", "jejunum", "ileum", "cecum", "colon", "rectum"]
N_SEG = len(SEGMENTS)

# Microbial load by segment, relative to the caecum. Density rises sharply into
# the large intestine; the small intestine carries orders of magnitude less.
SEG_MICROBES = np.array([0.001, 0.005, 0.05, 1.00, 0.80, 0.40])

# Oxygen availability by segment, relative. The paper reports that microbial
# oxidation of BBN to BCPN proceeds under aerobic and microaerobic but NOT
# anaerobic conditions, and the lumen grows more anaerobic distally. This
# weighting is what makes the caecum the dominant site of conversion rather
# than simply the site with the most bacteria.
SEG_OXYGEN = np.array([1.00, 0.80, 0.60, 0.45, 0.25, 0.15])

# Luminal volumes, mL (mouse).
SEG_VOL = np.array([0.10, 0.30, 0.30, 0.50, 0.30, 0.10])

# Segment emptying rate constants, /h. Transit is NOT uniform: small-intestinal
# passage takes a few hours while caecal and colonic residence runs to most of a
# day. This matters more than it looks. A poorly absorbed luminal metabolite can
# only make a large contribution to systemic exposure if it sits where it is
# made for long enough to be taken up -- caecal residence time, not absorption
# rate, is what lets microbial BCPN dominate urothelial exposure. Modelling the
# gut with one transit constant silently caps the microbial branch.
SEG_KTRANSIT = np.array([3.00, 1.20, 0.80, 0.10, 0.14, 0.50])

# Relative absorptive capacity by segment. The small intestine is the absorptive
# organ; colonic uptake of a neutral small molecule is real but several-fold
# lower per unit lumen. Without this the enterohepatic loop is nearly lossless,
# because deconjugated parent sitting in a slow caecum gets reabsorbed almost
# quantitatively, and recycling amplification runs away.
SEG_ABSORB = np.array([1.00, 1.00, 0.90, 0.30, 0.22, 0.15])


def _idx() -> dict:
    ix, k = {}, 0
    for s in SEGMENTS:                      # gut lumen, per segment
        for sp in ("BBN", "gBBN", "BCPNm", "BCPNh"):
            ix[f"{sp}_{s}"] = k
            k += 1
    for name in (
        "BBN_p",        # plasma parent
        "BCPNh_p",      # plasma BCPN, HOST origin
        "BCPNm_p",      # plasma BCPN, MICROBIAL origin
        "BBN_L",        # liver parent
        "gBBN_bile",    # biliary glucuronide en route to duodenum
        # Kidney tissue, interposed between plasma and bladder lumen. It is NOT
        # cosmetic: MTBLS3581 samples kidney in all six arms (27/26/12/12/16/16)
        # whereas urine exists in only two, so the kidney is the elimination-limb
        # compartment that actually has cross-arm coverage. Interposing it is
        # steady-state neutral by construction -- at steady state
        # k_kidney_out * K = k_renal * P, so every urinary and AUC quantity is
        # unchanged and only the transient and the new observable are added.
        "BCPNh_k",      # kidney tissue, host origin
        "BCPNm_k",      # kidney tissue, microbial origin
        "BCPNh_u",      # bladder lumen (urine), host origin
        "BCPNm_u",      # bladder lumen (urine), microbial origin
        "AUCh_t",       # cumulative urothelial exposure, host origin
        "AUCm_t",       # cumulative urothelial exposure, microbial origin
        "Uh",           # cumulative urinary excretion, host origin
        "Um",           # cumulative urinary excretion, microbial origin
    ):
        ix[name] = k
        k += 1
    return ix


IDX = _idx()
N_STATE = len(IDX)
SEG_BBN = np.array([IDX[f"BBN_{s}"] for s in SEGMENTS])
SEG_GBBN = np.array([IDX[f"gBBN_{s}"] for s in SEGMENTS])
SEG_BCPNM = np.array([IDX[f"BCPNm_{s}"] for s in SEGMENTS])
# Luminal BCPN of HOST origin, reaching the lumen by back-diffusion from plasma.
# Tracking it separately is not bookkeeping fussiness: without it the lumen has
# no host-derived BCPN at all, and the lumen/plasma ratio would discriminate by
# construction rather than by mechanism. The null has to be able to put BCPN in
# the gut, or beating it proves nothing.
SEG_BCPNH = np.array([IDX[f"BCPNh_{s}"] for s in SEGMENTS])


@dataclass(frozen=True)
class Params:
    """Illustrative magnitudes in absolute units. Not fitted -- see the module
    docstring and ``fit/PROVENANCE.md``."""

    # --- dosing -----------------------------------------------------------
    # Drinking-water mass balance for the source regimen, not a scaled cartoon.
    # The paper writes "0.05% BBN" / sometimes "0.05% v/v"; treated here as
    # mass/volume (0.5 mg/mL). A mouse drinks ~5 mL/day = 2.5 mg/day; BBN
    # (C8H18N2O2) is 174.244 g/mol, so 2.5 mg/day = 14.35 umol/day = 598 nmol/h.
    # An earlier release ran at 0.60 nmol/h -- ~1000x low -- which made the
    # "absolute units" framing untrue even though every ratio was unaffected.
    dose_rate: float = 598.0     # nmol/h BBN entering the duodenum (drinking water)

    # --- volumes and flows -------------------------------------------------
    V_plasma: float = 1.50       # mL
    V_bladder: float = 0.15      # mL
    V_kidney: float = 0.35       # mL, both mouse kidneys
    # Bile FLOW, not a volume. It converts the biliary glucuronide flux into the
    # concentration an assay would report on aspirated bile, which is the
    # observable MTBLS3581 carries for the two consortium arms (16 acquisition records each).
    Q_bile: float = 0.06         # mL/h bile flow into the duodenum

    # --- absorption --------------------------------------------------------
    ka_BBN: float = 0.50         # /h, parent absorbed from lumen to plasma
    # Ionisation of the carboxylic-acid metabolite motivates a lower assumed
    # absorption rate than the neutral parent. The magnitude is assigned, not
    # measured, and ionisation alone does not determine permeability.
    ka_BCPN: float = 0.030       # /h, BCPN absorbed from lumen to plasma
    k_back: float = 0.004        # /h, plasma -> lumen back-diffusion of BCPN

    # --- host hepatic metabolism ------------------------------------------
    k_uptake: float = 1.20       # /h, plasma BBN -> liver
    # Host omega-oxidation is kept LOW relative to glucuronidation because the
    # germ-free arm must produce little BCPN -- that is the paper's central
    # observation, and a model in which the liver is a major BCPN source cannot
    # reproduce it under any microbial parameterisation.
    k_ox_host: float = 0.020     # /h, hepatic omega-oxidation BBN -> BCPN_host
    k_gluc: float = 0.320        # /h, hepatic glucuronidation BBN -> gBBN
    k_bile: float = 0.900        # /h, gBBN -> bile
    # Hepatobiliary secretion of BCPN ITSELF into the duodenum, as distinct from
    # biliary export of the parent's glucuronide above. DEFAULT ZERO, and that
    # default is a structural hypothesis rather than a measurement: it is the
    # assumption on which V7 rests, so it is a named parameter that can be swept
    # instead of an omission nobody can see. See `nulls.biliary_escape` and
    # FINDINGS.md Pass 6. Applies to both provenance pools -- a transporter does
    # not know where the molecule was made.
    k_bile_BCPN: float = 0.0     # /h, plasma BCPN -> bile -> duodenal lumen

    # --- microbial reactions (gut lumen) ----------------------------------
    # Vmax AND Km are chosen, not measured. They are stated as a pair on
    # purpose: the model is exactly homogeneous under a joint rescaling of
    # (dose_rate, Vmax, Km), because lam*V * lam*c / (lam*K + lam*c) =
    # lam * V*c/(K+c) and every other term is linear. So this pair sets the
    # ABSOLUTE scale while every ratio, fraction and ordering in the build is
    # invariant to it. Km near 3-5 umol/mL keeps the luminal system
    # sub-saturating at the real dose (caecal BBN ~0.9 umol/mL), which is the
    # regime the structural argument was developed in.
    Vmax_gus: float = 9000.0     # nmol/h/mL, beta-glucuronidase, caecal scale
    Km_gus: float = 5000.0       # nmol/mL
    Vmax_ox: float = 9000.0      # nmol/h/mL, microbial oxidation BBN -> BCPN
    Km_ox: float = 3000.0        # nmol/mL

    # --- disposition and elimination --------------------------------------
    k_renal: float = 0.85        # /h, plasma BCPN -> kidney tissue
    k_kidney_out: float = 6.00   # /h, kidney tissue -> bladder lumen
    k_void: float = 0.35         # /h, bladder emptying
    # DIMENSIONLESS, and NOT an independent endpoint. `AUC_uro` integrates
    # k_uro * (bladder concentration); with k_uro dimensionless the integral
    # carries concentration x time, as an exposure index should. Earlier
    # releases labelled it h^-1 in the manuscript parameter table, which made
    # the printed "AUC" dimensionally a concentration.
    #
    # More important than the units: under these equations the index is EXACTLY
    # proportional to cumulative urinary excretion,
    #     AUC_uro / U_total == k_uro / (V_bladder * k_void)
    # in every arm, to machine precision (:func:`auc_is_urine_rescaled`, and a
    # regression asserts it). There is no urothelial compartment and nothing is
    # retained in tissue, so the index supplies NO information beyond `U_total`
    # and no host-response evidence at all. Read it as an assumed index.
    k_uro: float = 0.040         # dimensionless exposure-index scaling
    k_clear_p: float = 0.120     # /h, other plasma losses of BCPN
    k_clear_BBN: float = 0.060   # /h, other plasma losses of BBN


@dataclass(frozen=True)
class Arm:
    """A microbiome state. Multipliers act on the two microbial capacities.

    Splitting deconjugation from oxidation is what lets the two-member
    consortium be represented honestly: it carries beta-glucuronidase but cannot
    oxidise, so it must deconjugate gBBN without producing any BCPN.
    """
    key: str
    label: str
    gus: float = 1.0             # beta-glucuronidase capacity
    ox: float = 1.0              # microbial oxidation capacity
    plotted: bool = False
    # NOT "fitted". No arm in this build is fitted to anything -- the MAF
    # carries no values (fit/PROVENANCE.md), so every capacity here is
    # STRUCTURALLY ASSIGNED, not estimated. `reference` marks the arms whose
    # assignment defines the scale (conventional = 1, germ-free = 0) or is
    # taken as given (antibiotics); the rest had no part in assigning those.
    # NOT "held out": nothing here is calibrated, so there is no fit from which
    # an arm could be withheld and these are not an independent validation set.
    reference: bool = False


def michaelis(conc: np.ndarray, vmax: float, km: float) -> np.ndarray:
    c = np.maximum(conc, 0.0)
    return vmax * c / (km + c)


def rhs(t: float, y: np.ndarray, p: Params, arm: Arm) -> np.ndarray:
    dy = np.zeros_like(y)

    bbn = y[SEG_BBN]
    gbbn = y[SEG_GBBN]
    bcpnm = y[SEG_BCPNM]
    bcpnh = y[SEG_BCPNH]
    BBN_p, BCPNh_p, BCPNm_p = y[IDX["BBN_p"]], y[IDX["BCPNh_p"]], y[IDX["BCPNm_p"]]
    BBN_L, gBBN_bile = y[IDX["BBN_L"]], y[IDX["gBBN_bile"]]
    BCPNh_k, BCPNm_k = y[IDX["BCPNh_k"]], y[IDX["BCPNm_k"]]
    BCPNh_u, BCPNm_u = y[IDX["BCPNh_u"]], y[IDX["BCPNm_u"]]

    # concentrations in the lumen
    c_bbn = bbn / SEG_VOL
    c_gbbn = gbbn / SEG_VOL
    c_bcpnm = bcpnm / SEG_VOL
    c_bcpnh = bcpnh / SEG_VOL

    # --- microbial reactions ------------------------------------------------
    # capacity scales with microbial load; oxidation additionally with oxygen
    cap_gus = SEG_MICROBES * arm.gus
    cap_ox = SEG_MICROBES * SEG_OXYGEN * arm.ox

    v_deconj = michaelis(c_gbbn, p.Vmax_gus, p.Km_gus) * cap_gus * SEG_VOL
    v_micro_ox = michaelis(c_bbn, p.Vmax_ox, p.Km_ox) * cap_ox * SEG_VOL

    # --- luminal transit ----------------------------------------------------
    # first-order flow proximal -> distal; the distal segment empties as faeces
    def transit(x: np.ndarray) -> np.ndarray:
        rate = SEG_KTRANSIT * x
        out = -rate.copy()
        out[1:] += rate[:-1]           # what leaves segment i enters i+1
        return out                     # the distal segment empties as faeces

    # --- gut lumen ----------------------------------------------------------
    ka_seg = p.ka_BBN * SEG_ABSORB
    dy[SEG_BBN] = (transit(bbn)
                   - ka_seg * bbn
                   + v_deconj                       # gBBN -> BBN, microbial
                   - v_micro_ox)
    dy[SEG_GBBN] = transit(gbbn) - v_deconj
    # Back-diffusion from plasma is distributed over the lumen in proportion to
    # segment volume, and applies to BOTH provenance pools symmetrically -- the
    # molecule does not know where it was made.
    vfrac = SEG_VOL / SEG_VOL.sum()
    dy[SEG_BCPNM] = (transit(bcpnm)
                     + v_micro_ox
                     - p.ka_BCPN * SEG_ABSORB * bcpnm
                     + p.k_back * BCPNm_p * vfrac)
    dy[SEG_BCPNH] = (transit(bcpnh)
                     - p.ka_BCPN * SEG_ABSORB * bcpnh
                     + p.k_back * BCPNh_p * vfrac)

    # oral dose and bile both enter the duodenum
    dy[SEG_BBN[0]] += p.dose_rate
    dy[SEG_GBBN[0]] += p.k_bile * gBBN_bile

    # Hepatobiliary secretion of BCPN itself, delivered to the duodenum. Zero by
    # default. This route and independently assigned exchange coefficients can
    # raise host-only luminal BCPN above plasma; no universal bound is enforced.
    bile_bcpn_m = p.k_bile_BCPN * BCPNm_p
    bile_bcpn_h = p.k_bile_BCPN * BCPNh_p
    dy[SEG_BCPNM[0]] += bile_bcpn_m
    dy[SEG_BCPNH[0]] += bile_bcpn_h

    # --- systemic -----------------------------------------------------------
    abs_bbn = float(np.sum(ka_seg * bbn))
    abs_bcpnm = float(np.sum(p.ka_BCPN * SEG_ABSORB * bcpnm))
    abs_bcpnh = float(np.sum(p.ka_BCPN * SEG_ABSORB * bcpnh))
    back_m = p.k_back * BCPNm_p
    back_h = p.k_back * BCPNh_p

    dy[IDX["BBN_p"]] = (abs_bbn - p.k_uptake * BBN_p - p.k_clear_BBN * BBN_p)
    dy[IDX["BBN_L"]] = (p.k_uptake * BBN_p
                        - p.k_ox_host * BBN_L
                        - p.k_gluc * BBN_L)
    dy[IDX["gBBN_bile"]] = p.k_gluc * BBN_L - p.k_bile * gBBN_bile

    # host-origin BCPN enters plasma from the liver; microbial-origin from the gut
    dy[IDX["BCPNh_p"]] = (p.k_ox_host * BBN_L
                          + abs_bcpnh                 # reabsorbed after back-diffusion
                          - p.k_renal * BCPNh_p
                          - p.k_clear_p * BCPNh_p
                          - back_h
                          - bile_bcpn_h)
    dy[IDX["BCPNm_p"]] = (abs_bcpnm
                          - p.k_renal * BCPNm_p
                          - p.k_clear_p * BCPNm_p
                          - back_m
                          - bile_bcpn_m)

    dy[IDX["BCPNh_k"]] = p.k_renal * BCPNh_p - p.k_kidney_out * BCPNh_k
    dy[IDX["BCPNm_k"]] = p.k_renal * BCPNm_p - p.k_kidney_out * BCPNm_k

    dy[IDX["BCPNh_u"]] = p.k_kidney_out * BCPNh_k - p.k_void * BCPNh_u
    dy[IDX["BCPNm_u"]] = p.k_kidney_out * BCPNm_k - p.k_void * BCPNm_u

    dy[IDX["AUCh_t"]] = p.k_uro * BCPNh_u / p.V_bladder
    dy[IDX["AUCm_t"]] = p.k_uro * BCPNm_u / p.V_bladder
    dy[IDX["Uh"]] = p.k_void * BCPNh_u
    dy[IDX["Um"]] = p.k_void * BCPNm_u

    return dy


def simulate(p: Params, arm: Arm, t: np.ndarray) -> np.ndarray:
    y0 = np.zeros(N_STATE)
    sol = solve_ivp(rhs, (float(t[0]), float(t[-1])), y0, t_eval=t,
                    args=(p, arm), method="LSODA", rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    return sol.y


def trace(y: np.ndarray, p: Params) -> dict:
    """Observables, split into what is latent and what an assay could report.

    ``bcpn_*`` quantities are the SUM of the two provenance pools -- they are
    what a mass spectrometer sees. The separate pools are latent, and are
    reported only so the model can be interrogated about them.
    """
    out = {}
    for i, s in enumerate(SEGMENTS):
        out[f"c_BBN_{s}"] = y[SEG_BBN[i]] / SEG_VOL[i]
        out[f"c_gBBN_{s}"] = y[SEG_GBBN[i]] / SEG_VOL[i]
        out[f"c_BCPNm_{s}"] = y[SEG_BCPNM[i]] / SEG_VOL[i]     # latent
        out[f"c_BCPNh_{s}"] = y[SEG_BCPNH[i]] / SEG_VOL[i]     # latent
        out[f"c_BCPN_{s}"] = out[f"c_BCPNm_{s}"] + out[f"c_BCPNh_{s}"]   # measurable
    out["c_BBN_plasma"] = y[IDX["BBN_p"]] / p.V_plasma
    out["c_BCPNh_plasma"] = y[IDX["BCPNh_p"]] / p.V_plasma    # latent
    out["c_BCPNm_plasma"] = y[IDX["BCPNm_p"]] / p.V_plasma    # latent
    out["c_BCPN_plasma"] = out["c_BCPNh_plasma"] + out["c_BCPNm_plasma"]
    out["c_BCPNh_kidney"] = y[IDX["BCPNh_k"]] / p.V_kidney            # latent
    out["c_BCPNm_kidney"] = y[IDX["BCPNm_k"]] / p.V_kidney            # latent
    out["c_BCPN_kidney"] = out["c_BCPNh_kidney"] + out["c_BCPNm_kidney"]
    # Concentration in bile as aspirated: biliary flux divided by bile flow.
    out["c_gBBN_bile"] = p.k_bile * y[IDX["gBBN_bile"]] / p.Q_bile
    out["c_BCPN_urine"] = (y[IDX["BCPNh_u"]] + y[IDX["BCPNm_u"]]) / p.V_bladder
    out["U_total"] = y[IDX["Uh"]] + y[IDX["Um"]]
    out["AUC_uro"] = y[IDX["AUCh_t"]] + y[IDX["AUCm_t"]]
    out["AUC_uro_host"] = y[IDX["AUCh_t"]]
    out["AUC_uro_micro"] = y[IDX["AUCm_t"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["micro_fraction"] = np.divide(
            y[IDX["AUCm_t"]], y[IDX["AUCh_t"]] + y[IDX["AUCm_t"]],
            out=np.zeros_like(y[IDX["AUCm_t"]]),
            where=(y[IDX["AUCh_t"]] + y[IDX["AUCm_t"]]) > 1e-12)
    return out


def auc_is_urine_rescaled(p: Params) -> float:
    """The constant by which ``AUC_uro`` is cumulative urinary excretion.

    ``dAUC/dt = k_uro * A_u / V_bladder`` and ``dU/dt = k_void * A_u``, so the
    two integrals differ by a fixed factor and the exposure index carries no
    information that ``U_total`` does not. Stated as a function, not a comment,
    so the manuscript can print the identity and a regression can assert it.
    """
    return p.k_uro / (p.V_bladder * p.k_void)


def lumen_plasma_ratio(y: np.ndarray, p: Params, segment: str = "cecum") -> float:
    """Candidate terminal compartment ratio, summing latent provenance pools.

    A source contrast under specified parameters, not a unique origin marker.
    Exchange coefficients are independent; no passive reciprocity constraint
    is enforced. Biliary delivery and exchange changes can raise a host-only
    ratio above one. Empirical ratios additionally require comparable matrix
    normalisation before a physical unity threshold can be interpreted.
    """
    i = SEGMENTS.index(segment)
    c_lum = (y[SEG_BCPNM[i], -1] + y[SEG_BCPNH[i], -1]) / SEG_VOL[i]
    c_pl = (y[IDX["BCPNh_p"], -1] + y[IDX["BCPNm_p"], -1]) / p.V_plasma
    return float(c_lum / c_pl) if c_pl > 1e-15 else np.inf


def param_names() -> list[str]:
    return [f.name for f in fields(Params)]
