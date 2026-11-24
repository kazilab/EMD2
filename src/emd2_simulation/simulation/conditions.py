"""Assigned microbial capacities for the six model arms."""
from .model import Arm

ARMS = [
    Arm("B", "Conventional", gus=1.00, ox=1.00, plotted=True, reference=True),
    Arm("BA", "Antibiotics", gus=0.05, ox=0.05, plotted=True, reference=True),
    Arm("Germ-Free", "Germ-free", gus=0.00, ox=0.00, plotted=True, reference=True),
    Arm("Monocolonized", "Mono-colonised", gus=0.70, ox=0.70),
    Arm("2Mem", "Two-member (GUS, no oxidase)", gus=1.00, ox=0.00, plotted=True),
    Arm("3Mem", "Three-member", gus=0.90, ox=0.75),
]
BY_KEY = {a.key: a for a in ARMS}
PLOTTED = [a for a in ARMS if a.plotted]
NON_REFERENCE = [a for a in ARMS if not a.reference]

# Candidate compartment for the lumen/plasma ratio.
RATIO_SEGMENT = "cecum"

