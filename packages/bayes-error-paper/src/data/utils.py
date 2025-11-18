from pathlib import Path
import numpy as np
import requests
from tqdm import tqdm

data_cache_dir = Path('./.data_cache')

def download_if_not_exists(url, path):
    path = Path(path)
    if not path.exists():
        print(f'Downloading dataset from {url}...')
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total = int(response.headers.get('content-length', 0))
        path.parent.mkdir(parents=True, exist_ok=True)
        with (
            open(path, 'wb') as f,
            tqdm(
                total=total or None,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def binarize_hard_labels(labels, positive_class_indices):
    return np.where(np.isin(labels, positive_class_indices), 1, 0)


def binarize_soft_labels(soft_labels, positive_class_indices):
    return soft_labels[:, positive_class_indices].sum(axis=1) / soft_labels.sum(
        axis=1
    )


def generate_approximate_soft_labels(rng, soft_labels_clean, n_hard_labels):
    soft_labels_hard = (
        rng.binomial(n_hard_labels, soft_labels_clean) / n_hard_labels
    )
    return soft_labels_hard
