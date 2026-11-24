"""Integrate the registered EMD2 treatment arms."""
import numpy as np
from .model import Params, simulate, trace, lumen_plasma_ratio
from .conditions import ARMS, RATIO_SEGMENT

def run_arms(p: Params, t: np.ndarray) -> dict:
    out = {}
    for a in ARMS:
        y = simulate(p, a, t)
        tr = trace(y, p)
        tr["ratio_cecum"] = lumen_plasma_ratio(y, p, RATIO_SEGMENT)
        out[a.key] = tr
    return out
