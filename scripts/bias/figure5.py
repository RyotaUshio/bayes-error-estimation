"""Figure 5: The graph of the function $L_{Err}(q)$."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def L_Err(q):
    """The function L_Err(q) defined in Equation (4)."""
    return q * (1 - q) / np.abs(2 * q - 1)

plt.rcParams.update({
    "font.size": 12,
})

q = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
ax.plot(q, L_Err(q))

ax.set_xlabel('$q$')
ax.set_ylabel(r'$L_{\mathrm{Err}}(q)$')
ax.set_xlim(0, 1)
ax.set_ylim(0, 10)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

outfile = Path("results/graph-of-L_Err.pdf")
outfile.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(outfile, bbox_inches='tight')
print(f"Saved to {outfile}")
