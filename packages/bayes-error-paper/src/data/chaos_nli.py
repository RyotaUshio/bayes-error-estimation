import json
import subprocess
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from .utils import binarize_hard_labels, binarize_soft_labels, data_cache_dir

type DatasetName = Literal['snli', 'mnli', 'abduptive_nli']

dataset_dir = data_cache_dir / 'chaos-nli'
dataset_zip_path = dataset_dir / 'chaosNLI_v1.0.zip'
predictions_zip_path = dataset_dir / 'model_predictions.zip'
dataset_filenames: dict[DatasetName, str] = {
    'snli': 'chaosNLI_v1.0/chaosNLI_snli.jsonl',
    'mnli': 'chaosNLI_v1.0/chaosNLI_mnli_m.jsonl',
    'abduptive_nli': 'chaosNLI_v1.0/chaosNLI_alphanli.jsonl',
}
predictions_filenames: dict[DatasetName, str] = {
    'snli': 'model_predictions/model_predictions_for_snli_mnli.json',
    'mnli': 'model_predictions/model_predictions_for_snli_mnli.json',
    'abduptive_nli': 'model_predictions/model_predictions_for_abdnli.json',
}
classes: dict[DatasetName, list[str | int]] = {
    'snli': ['e', 'n', 'c'],
    'mnli': ['e', 'n', 'c'],
    'abduptive_nli': [1, 2],
}
positive_class_indices: dict[DatasetName, list[int]] = {
    'snli': [0, 1],
    'mnli': [0, 1],
    'abduptive_nli': [1],
}


def download_if_not_exists(url: str, download_path: Path):
    if download_path.exists():
        return

    try:
        download_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['wget', url, '-O', download_path], check=True)
    except Exception as e:
        print(f'Error downloading from {url}: {e}')


def download_chaos_nli_if_not_exists():
    download_if_not_exists(
        'https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip',
        dataset_zip_path,
    )
    download_if_not_exists(
        'https://www.dropbox.com/s/qy7uk6ajm5x6dl6/model_predictions.zip',
        predictions_zip_path,
    )


def extract_if_not_exists(filename: str):
    jsonl_path = dataset_dir / filename
    if jsonl_path.exists():
        return jsonl_path

    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        jsonl_path = zip_ref.extract(filename, dataset_dir)
        return Path(jsonl_path)


def load_chaos_nli_from_file(dataset_name: DatasetName):
    dataset_path = extract_if_not_exists(dataset_filenames[dataset_name])
    predictions_path = extract_if_not_exists(
        predictions_filenames[dataset_name]
    )
    with open(predictions_path, 'r') as f:
        raw_predictions = json.load(f)

    soft_labels = []
    labels = []
    model_predictions = {}

    with open(dataset_path, 'r') as f:
        # The dataset is in JSONL format, which is a line-by-line JSON object
        for line in f:
            instance = json.loads(line)

            soft_label = instance['label_count']
            soft_labels.append(soft_label)

            label = classes[dataset_name].index(instance['old_label'])
            labels.append(label)

            uid = instance['uid']
            for model_name, predictions in raw_predictions.items():
                logits = predictions[uid]['logits']
                prediction = np.argmax(logits)

                if model_name not in model_predictions:
                    model_predictions[model_name] = []
                model_predictions[model_name].append(prediction)

    positives = positive_class_indices[dataset_name]

    soft_labels = binarize_soft_labels(np.array(soft_labels), positives)
    labels = binarize_hard_labels(np.array(labels), positives)
    sota_error = np.min(
        [
            (
                labels != binarize_hard_labels(np.array(predictions), positives)
            ).mean()
            for predictions in model_predictions.values()
        ]
    )

    return soft_labels, labels, sota_error


class ChaosNliOptions(BaseModel):
    dataset: DatasetName


def load_chaos_nli(options: ChaosNliOptions):
    download_chaos_nli_if_not_exists()
    soft_labels, labels, sota_error = load_chaos_nli_from_file(options.dataset)

    return {
        'corrupted': {
            'soft_labels': soft_labels,
            'labels': labels,
        },
    }


# if __name__ == '__main__':
#     snli = load_chaos_nli('snli')
#     mnli = load_chaos_nli('mnli')
#     abduptive_nli = load_chaos_nli('abduptive_nli')

#     print(f'SOTA error for SNLI: {snli["sota_error"]}')
#     print(f'SOTA error for MNLI: {mnli["sota_error"]}')
#     print(f'SOTA error for Abduptive NLI: {abduptive_nli["sota_error"]}')
