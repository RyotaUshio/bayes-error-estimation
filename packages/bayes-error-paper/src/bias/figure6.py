from pathlib import Path
import matplotlib.pyplot as plt
from ..data.cifar10 import load_cifar10h_soft_labels


def main():
    data = load_cifar10h_soft_labels()
    plt.rcParams['font.size'] = 12
    plt.hist(data, bins=5, rwidth=0.5)
    plt.xlabel('Soft labels for the positive class')
    plt.ylabel('Number of samples')
    plt.xticks(
        [0.1, 0.3, 0.5, 0.7, 0.9],
        ['0–0.2', '0.2–0.4', '0.4–0.6', '0.6–0.8', '0.8–1.0'],
    )
    plt.tick_params(
        axis='x', which='both', bottom=False, top=False, labelbottom=True
    )

    outfile = Path('results/eta_dist.pdf')
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, bbox_inches='tight')
    print(f'Saved to {outfile}')
