"""Compartment model of BBN disposition and BCPN of host or microbial origin.

Amounts are in nmol, volumes in mL and time in hours. Model kinetics are assigned.
Assays measure total BCPN across both origins. The source-resolved pools are latent.
The exposure index is proportional to cumulative urinary excretion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, fields

import numpy as np
from scipy.integrate import solve_ivp

# Gut compartments in proximal-to-distal order, using MTBLS3581 specimen labels.
SEGMENTS = ["duodenum", "jejunum", "ileum", "cecum", "colon", "rectum"]
N_SEG = len(SEGMENTS)

# Assigned microbial load relative to the caecum.
SEG_MICROBES = np.array([0.001, 0.005, 0.05, 1.00, 0.80, 0.40])

# Relative oxygen availability by segment.
SEG_OXYGEN = np.array([1.00, 0.80, 0.60, 0.45, 0.25, 0.15])

# Luminal volumes, mL (mouse).
SEG_VOL = np.array([0.10, 0.30, 0.30, 0.50, 0.30, 0.10])

# First-order emptying rates (per hour) from each segment.
SEG_KTRANSIT = np.array([3.00, 1.20, 0.80, 0.10, 0.14, 0.50])

# Relative absorptive capacity by segment.
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
        # At steady state, renal inflow from plasma equals outflow to the bladder.
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
# Host-derived BCPN can reach the lumen from plasma or bile.
SEG_BCPNH = np.array([IDX[f"BCPNh_{s}"] for s in SEGMENTS])


@dataclass(frozen=True)
class Params:
    """Assigned kinetic parameters, compartment volumes and flows."""

    # The dose estimate assumes the reported drinking-water concentration is w/v.
    dose_rate: float = 598.0     # nmol/h BBN entering the duodenum (drinking water)

    # --- volumes and flows -------------------------------------------------
    V_plasma: float = 1.50       # mL
    V_bladder: float = 0.15      # mL
    V_kidney: float = 0.35       # mL, both mouse kidneys
    # Dividing biliary flux by flow gives the concentration in aspirated bile.
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
    # Hepatic oxidation is assigned a lower rate than glucuronidation.
    k_ox_host: float = 0.020     # /h, hepatic omega-oxidation BBN -> BCPN_host
    k_gluc: float = 0.320        # /h, hepatic glucuronidation BBN -> gBBN
    k_bile: float = 0.900        # /h, gBBN -> bile
    # Biliary BCPN secretion has an assumed zero baseline for both source pools.
    k_bile_BCPN: float = 0.0     # /h, plasma BCPN -> bile -> duodenal lumen

    # --- microbial reactions (gut lumen) ----------------------------------
    # Jointly scaling dose, Vmax and Km preserves the model concentration ratios.
    Vmax_gus: float = 9000.0     # nmol/h/mL, beta-glucuronidase, caecal scale
    Km_gus: float = 5000.0       # nmol/mL
    Vmax_ox: float = 9000.0      # nmol/h/mL, microbial oxidation BBN -> BCPN
    Km_ox: float = 3000.0        # nmol/mL

    # --- disposition and elimination --------------------------------------
    k_renal: float = 0.85        # /h, plasma BCPN -> kidney tissue
    k_kidney_out: float = 6.00   # /h, kidney tissue -> bladder lumen
    k_void: float = 0.35         # /h, bladder emptying
    # AUC_uro / U_total equals k_uro / (V_bladder * k_void) in every arm.
    k_uro: float = 0.040         # dimensionless exposure-index scaling
    k_clear_p: float = 0.120     # /h, other plasma losses of BCPN
    k_clear_BBN: float = 0.060   # /h, other plasma losses of BBN


@dataclass(frozen=True)
class Arm:
    """A microbiome state with independent capacity multipliers."""
    key: str
    label: str
    gus: float = 1.0             # beta-glucuronidase capacity
    ox: float = 1.0              # microbial oxidation capacity
    plotted: bool = False
    # Marks the conventional, antibiotic and germ-free reference scenarios.
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

    # First-order transit proceeds distally, ending in faecal loss.
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
    # Both source pools distribute plasma back-diffusion by segment volume.
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

    # Biliary BCPN enters the duodenum.
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
    """Return compartment totals and the latent source-specific trajectories."""
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
    """Return the proportionality factor between AUC_uro and U_total."""
    return p.k_uro / (p.V_bladder * p.k_void)


def lumen_plasma_ratio(y: np.ndarray, p: Params, segment: str = "cecum") -> float:
    """Terminal ratio of total luminal to plasma BCPN.

    Independent exchange coefficients and biliary transport can produce
    host-only ratios above one. Interpreting a measured cross-matrix ratio
    requires comparable assay normalisation.
    """
    i = SEGMENTS.index(segment)
    c_lum = (y[SEG_BCPNM[i], -1] + y[SEG_BCPNH[i], -1]) / SEG_VOL[i]
    c_pl = (y[IDX["BCPNh_p"], -1] + y[IDX["BCPNm_p"], -1]) / p.V_plasma
    return float(c_lum / c_pl) if c_pl > 1e-15 else np.inf


def param_names() -> list[str]:
    return [f.name for f in fields(Params)]
