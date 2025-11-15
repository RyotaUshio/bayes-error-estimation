"""Figure 2: A comparison of our bias bound (Corollary 2) and the existing bound by Ishida et al. (2023) with n=10000, m = 50."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from bias_bounds import B_inner, existing_bias_bound  # ty: ignore[unresolved-import]

m = 50
E_list = np.logspace(-4, np.log10(0.5), num=100, base=10)
t_list = np.linspace(1e-10, 0.5, num=1000, endpoint=False)

ours = np.array([B_inner(E=E, m=m, t=t_list) for E in E_list]).min(axis=1)
existing = existing_bias_bound(n=10000, m=m)

# Plot results
plt.rcParams.update(
    {
        'font.size': 16,
    }
)
fig, ax = plt.subplots()
ax.plot(E_list, ours, label='Our result ($B(E, m)$)')
ax.plot(E_list, [existing] * len(E_list), label='Ishida et al. (2023)')

ax.set_xlabel('$E$')
ax.set_ylabel('Maximum bias')

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlim(E_list.min(), E_list.max())
ax.set_ylim(10e-4, 1)

# Indicate the test error of ViT as a vertical, black dashed line
E_ViT = 0.05 * 0.01
ax.vlines(
    E_ViT, 0, 1, colors='black', linestyles='dashed', linewidth=1, alpha=0.8
)

fig.legend(loc='lower right', bbox_to_anchor=(0.89, 0.12))

outfile = Path('results/bias-bounds.pdf')
outfile.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(outfile, bbox_inches='tight')
print(f'Saved to {outfile}')
