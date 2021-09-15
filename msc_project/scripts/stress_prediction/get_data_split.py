import os

import pandas as pd

import wandb

from msc_project.constants import (
    STRESS_PREDICTION_PROJECT_NAME,
    SheetNames,
    PREPROCESSED_DATA_ARTIFACT,
    BASE_DIR,
)
from msc_project.scripts.get_data_split import DatasetPreparer
from msc_project.scripts.hrv.get_hrv import get_artifact_dataframe


def get_autoencoder_preprocessed_data_artifact(
    stress_prediction_preprocessed_data_artifact,
):
    run = stress_prediction_preprocessed_data_artifact.logged_by()
    used_artifacts = run.used_artifacts()
    preprocessed_data_artifacts = [
        artifact
        for artifact in used_artifacts
        if artifact.type == PREPROCESSED_DATA_ARTIFACT
    ]
    assert len(preprocessed_data_artifacts) == 1
    return preprocessed_data_artifacts[0]


def handle_data_split(signals: pd.DataFrame, data_name: str):
    """
    Split data and save.

    :param signals:
    :param data_name: only_downsampled/labels/intermediate_preprocessed etc.
    :return:
    """
    dataset_preparer = DatasetPreparer(
        noise_tolerance=0,
        signals=signals,
        noisy_mask=noisy_mask,
    )
    train_signals, val_signals, noisy_signals = dataset_preparer.get_dataset()

    save_dir = os.path.join(run_dir, data_name)
    train_signals.to_pickle(os.path.join(save_dir, "train.pkl"))
    val_signals.to_pickle(os.path.join(save_dir, "val.pkl"))
    noisy_signals.to_pickle(os.path.join(save_dir, "noisy.pkl"))
    return train_signals, val_signals, noisy_signals


if __name__ == "__main__":
    run = wandb.init(
        project=STRESS_PREDICTION_PROJECT_NAME, job_type="data_split", save_code=True
    )
    sheet_name = SheetNames.INFINITY.value
    preprocessed_data_artifact_version = 1

    run_dir = os.path.join(
        BASE_DIR, "data", "stress_prediction", "data_split", sheet_name, run.name
    )
    os.makedirs(run_dir)

    stress_prediction_preprocessed_data_artifact = run.use_artifact(
        artifact_or_name=f"{sheet_name}:v{preprocessed_data_artifact_version}",
        type=PREPROCESSED_DATA_ARTIFACT,
    )

    autoencoder_preprocessed_data_artifact = get_autoencoder_preprocessed_data_artifact(
        stress_prediction_preprocessed_data_artifact
    )
    noisy_mask = get_artifact_dataframe(
        run=run,
        artifact_or_name=autoencoder_preprocessed_data_artifact,
        pkl_filename="windowed_noisy_mask.pkl",
    )

    only_downsampled_data = get_artifact_dataframe(
        run=run,
        artifact_or_name=stress_prediction_preprocessed_data_artifact,
        pkl_filename="only_downsampled_data.pkl",
    )

    dataset_preparer = DatasetPreparer(
        noise_tolerance=0,
        signals=only_downsampled_data,
        noisy_mask=noisy_mask,
    )
    train_signals, val_signals, noisy_signals = dataset_preparer.get_dataset()
