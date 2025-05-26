"""Figure 3: The corruption function f(p; a, b) with parameters"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def f(p, a, b):
    """The corruption function f(p; a, b) in defined in Section 4.1.

    This is the inverse function of the two-parameter beta calibration map (Kull et al., 2017)
    and can express various continuous increasing transformations on the interval (0, 1) 
    depending on the parameters a and b.
    """
    return 1 / (
        1 + ( (1 - p) / p )**(1/a) * ( (1 - b) / b )
    )

parser = argparse.ArgumentParser()
parser.add_argument('--a', type=float, default=2.0, help='Default: 2.0')
parser.add_argument('--b', type=float, default=0.7, help='Default: 0.7')
args = parser.parse_args()

plt.rcParams.update({
    "font.size": 16,
})
fig, ax = plt.subplots()

p = np.linspace(1e-10, 1 - 1e-10, 100)
ax.plot(p, f(p, args.a, args.b))

ax.set_xlabel("$p$")
ax.set_ylabel("$f(p; a, b)$")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")

outfile = Path(f"results/corruption_map_{args.a:.1f}_{args.b:.1f}.pdf")
outfile.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(outfile, bbox_inches="tight")
print(f"Saved to {outfile}")
