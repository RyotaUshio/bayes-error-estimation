from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path, help='Path to the JSON file generated by uv run scripts/bias/figure7_experiment.py [a, b, c]')
    parser.add_argument('-o', '--output', type=Path, help='Path to the output PDF file')
    parser.add_argument('-s', '--show', action='store_true')
    args = parser.parse_args()
    assert args.input.suffix == '.json'
    assert args.output is None or args.output.suffix == '.pdf'

    with open(args.input, 'r') as f:
        results = json.load(f)

    plt.rcParams["font.size"] = 14

    m, bias, theoretical = zip(*map(lambda x: x.values(), results.values()))
    plt.plot(m, bias)
    plt.plot(m, theoretical, 'k--')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((10, 1000))
    plt.ylim((1e-7, 1))
    plt.xlabel('$m$')
    plt.ylabel('Bias')
    plt.gca().set_aspect('equal')
    plt.tight_layout()

    outfile = args.output or args.input.with_suffix('.pdf')
    print(f'Saving to {outfile}')
    plt.savefig(outfile, bbox_inches='tight')

    if args.show:
        plt.show()
