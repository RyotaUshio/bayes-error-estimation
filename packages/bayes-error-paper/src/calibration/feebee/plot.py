import matplotlib.pyplot as plt

from .data import FeeBeeData


def plot(data: FeeBeeData) -> plt.Figure:
    plt.rcParams.update({'font.size': 16})

    order = [
        'isotonic',
        'hist10',
        'hist25',
        'hist50',
        'hist100',
        'beta',
        'beta-am',
        'beta-ab',
        'beta-a',
        'platt',
    ]

    scores = {
        name: data.results[name]['score_lower']
        + data.results[name]['score_upper']
        for name in order
        if name in data.results
    }

    min_score = min(scores.values())
    colors = [
        '#F28E2B' if score == min_score else '#4C72B0'
        for score in scores.values()
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(scores.keys()), list(scores.values()), color=colors)

    ax.set_ylabel('FeeBee Score')
    ax.set_ylim(0, max(scores.values()) * 1.05)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig
