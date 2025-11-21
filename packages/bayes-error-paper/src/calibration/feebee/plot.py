import matplotlib.pyplot as plt

from .data import FeeBeeData


def plot(data: FeeBeeData) -> plt.Figure:
    scores = {
        name: result['score_lower'] + result['score_upper']
        for name, result in data.results.items()
        if name != 'corrupted'
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(scores.keys(), scores.values(), color='#4C72B0')

    ax.set_ylabel('FeeBee Score')
    ax.set_ylim(0, max(scores.values()) * 1.05)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig
