import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_data(args):
    data = {"spearman_corr": [], "results": {}}

    order = [
        "clean",
        "hard",
        "corrupted",
        "isotonic",
        "hist10",
        "hist25",
        "hist50",
        "hist100",
        "beta",
    ]

    for file_path in args.input:
        with open(file_path, "r") as f:
            file_data = json.load(f)

        spearman_corr = file_data["metadata"]["spearman_corr"]
        data["spearman_corr"].append(spearman_corr)

        methods = [key for key in order if key in file_data["results"]]

        for method in methods:
            method_data = file_data["results"][method]

            if method not in data["results"]:
                data["results"][method] = {
                    "point_estimates": [],
                    "low_ci": [],
                    "high_ci": [],
                    "corrs": [],
                }

            data["results"][method]["point_estimates"].append(
                method_data["point_estimate"]
            )
            data["results"][method]["low_ci"].append(
                method_data["confidence_interval"]["low"]
            )
            data["results"][method]["high_ci"].append(
                method_data["confidence_interval"]["high"]
            )
            data["results"][method]["corrs"].append(spearman_corr)

    for method in data["results"]:
        sort_indices = np.argsort(data["results"][method]["corrs"])
        data["results"][method]["corrs"] = np.array(data["results"][method]["corrs"])[
            sort_indices
        ]
        data["results"][method]["point_estimates"] = np.array(
            data["results"][method]["point_estimates"]
        )[sort_indices]
        data["results"][method]["low_ci"] = np.array(data["results"][method]["low_ci"])[
            sort_indices
        ]
        data["results"][method]["high_ci"] = np.array(
            data["results"][method]["high_ci"]
        )[sort_indices]

    return data


def plot_results(data, args):
    omit = set(["clean", "hard"] + args.omit.split(","))

    plt.rcParams["font.size"] = 10
    fig, ax = plt.subplots()

    cmap = plt.get_cmap("Set2")
    methods = [key for key in data["results"] if key not in omit]
    colors = cmap(np.linspace(0, 1, len(methods)))
    for method, color in zip(methods, colors):
        method_data = data["results"][method]
        zorder = 1 if method == "isotonic" else 0

        if args.fancy_errorbar:
            ax.plot(
                method_data["corrs"],
                method_data["point_estimates"],
                marker="o",
                linestyle="-",
                color=color,
                label=method,
                zorder=zorder,
            )
            ax.fill_between(
                method_data["corrs"],
                method_data["low_ci"],
                method_data["high_ci"],
                color=color,
                alpha=0.2,
                zorder=zorder,
            )
        else:
            ax.errorbar(
                method_data["corrs"],
                method_data["point_estimates"],
                yerr=[
                    method_data["point_estimates"] - method_data["low_ci"],
                    method_data["high_ci"] - method_data["point_estimates"],
                ],
                fmt="o-",
                color=color,
                label=method,
                capsize=3,
                elinewidth=1,
                markersize=5,
                zorder=zorder,
            )

    clean_value = float(np.mean(data["results"]["clean"]["point_estimates"]))
    ax.axhline(
        y=clean_value,
        color="black",
        linestyle="--",
        linewidth=0.8,
        xmin=0,
        xmax=1,
        zorder=-1,
    )

    ylim = ax.get_ylim()
    ax.set_ylim(0, ylim[1])

    ax.set_xlabel("Spearman's rank correlation coefficient $\\rho$", fontsize=12)
    ax.set_ylabel("Estimated Bayes error (%)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="lower left", bbox_to_anchor=(1, 0))

    fig.tight_layout()

    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input files", nargs="+")
    parser.add_argument("-o", "--output", type=Path, help="Output file", required=True)
    parser.add_argument("-s", "--show", action="store_true")
    parser.add_argument("--omit", type=str, default="")
    parser.add_argument("--fancy_errorbar", action="store_true")
    args = parser.parse_args()

    data = load_data(args)
    fig, ax = plot_results(data, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Saved to {args.output}")

    if args.show:
        plt.show()
