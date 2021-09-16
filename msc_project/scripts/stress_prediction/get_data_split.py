"""
Split various datas (only downsampled, traditional preprocessed etc.) into train, val, noisy.
Save and upload split DataFrames.
"""
import os
from typing import Iterable

import matplotlib.pyplot as plt
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


def handle_data_split(
    signals: pd.DataFrame,
    data_name: str,
    noise_tolerance: float,
    validation_participants: Iterable[str] = None,
):
    """
    Split data and save.

    :param signals:
    :param data_name: only_downsampled/labels/intermediate_preprocessed etc.
    :return:
    """
    dataset_preparer = DatasetPreparer(
        noise_tolerance=noise_tolerance,
        signals=signals,
        noisy_mask=noisy_mask,
        validation_participants=validation_participants,
    )
    train_signals, val_signals, noisy_signals = dataset_preparer.get_dataset()

    save_dir = os.path.join(run_dir, data_name)
    os.makedirs(save_dir)
    train_signals.to_pickle(os.path.join(save_dir, "train.pkl"))
    val_signals.to_pickle(os.path.join(save_dir, "val.pkl"))
    noisy_signals.to_pickle(os.path.join(save_dir, "noisy.pkl"))
    return train_signals, val_signals, noisy_signals


if __name__ == "__main__":
    upload_artifact: bool = True
    validation_participants = [
        "0720202421P1_608",
        "0725095437P2_608",
        "0726094551P5_609",
        "0802131257P22_608",
        "0730114205P18_lamp",
    ]
    config = {"noise_tolerance": 1, "validation_participants": validation_participants}

    run = wandb.init(
        project=STRESS_PREDICTION_PROJECT_NAME,
        job_type="data_split",
        config=config,
        save_code=True,
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
    traditional_preprocessed_data = get_artifact_dataframe(
        run=run,
        artifact_or_name=stress_prediction_preprocessed_data_artifact,
        pkl_filename="traditional_preprocessed_data.pkl",
    )
    intermediate_preprocessed_data = get_artifact_dataframe(
        run=run,
        artifact_or_name=stress_prediction_preprocessed_data_artifact,
        pkl_filename="intermediate_preprocessed_data.pkl",
    )
    proposed_denoised_data = get_artifact_dataframe(
        run=run,
        artifact_or_name=stress_prediction_preprocessed_data_artifact,
        pkl_filename="proposed_denoised_data.pkl",
    )
    labels = get_artifact_dataframe(
        run=run,
        artifact_or_name=stress_prediction_preprocessed_data_artifact,
        pkl_filename="labels.pkl",
    )

    handle_data_split(
        signals=only_downsampled_data,
        data_name="only_downsampled",
        noise_tolerance=config["noise_tolerance"],
    )
    handle_data_split(
        signals=traditional_preprocessed_data,
        data_name="traditional_preprocessed",
        noise_tolerance=config["noise_tolerance"],
    )
    handle_data_split(
        signals=intermediate_preprocessed_data,
        data_name="intermediate_preprocessed",
        noise_tolerance=config["noise_tolerance"],
    )
    handle_data_split(
        signals=proposed_denoised_data,
        data_name="proposed_denoised",
        noise_tolerance=config["noise_tolerance"],
    )

    handle_data_split(
        signals=labels,
        data_name="labels",
        noise_tolerance=config["noise_tolerance"],
        validation_participants=validation_participants,
    )

    if upload_artifact:
        artifact_type = "data_split"
        artifact = wandb.Artifact(
            name=f"{sheet_name}_{artifact_type}", type=artifact_type
        )
        artifact.add_dir(run_dir)
        run.log_artifact(artifact, type=artifact_type)
    run.finish()

    noisy_proportions = noisy_mask.sum(axis=0) / noisy_mask.shape[0]
    means = noisy_proportions.groupby("participant").mean()
    means.plot.bar()
    plt.tight_layout()
    plt.show()
