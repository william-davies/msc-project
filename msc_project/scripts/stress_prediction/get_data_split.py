import os

import pandas as pd

import wandb

from msc_project.constants import (
    STRESS_PREDICTION_PROJECT_NAME,
    SheetNames,
    PREPROCESSED_DATA_ARTIFACT,
)
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


if __name__ == "__main__":
    run = wandb.init(
        project=STRESS_PREDICTION_PROJECT_NAME, job_type="data_split", save_code=True
    )
    sheet_name = SheetNames.INFINITY.value
    preprocessed_data_artifact_version = 1

    preprocessed_data_artifact = run.use_artifact(
        artifact_or_name=f"{sheet_name}:v{preprocessed_data_artifact_version}",
        type=PREPROCESSED_DATA_ARTIFACT,
    )

    only_downsampled_data = get_artifact_dataframe(
        run=run,
        artifact_or_name=preprocessed_data_artifact,
        pkl_filename="only_downsampled_data.pkl",
    )

    noisy_mask = pd.read_pickle(os.path.join(download_fp, "windowed_noisy_mask.pkl"))

    dataset_preparer = DatasetPreparer(
        noise_tolerance=metadata["noise_tolerance"],
        signals=intermediate_preprocessed_signals,
        noisy_mask=noisy_mask,
    )
    train_signals, val_signals, noisy_signals = dataset_preparer.get_dataset()
