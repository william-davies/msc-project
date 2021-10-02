"""
Load HRV features (input) and labels (target). Filter by clean signals.
"""
import os

import wandb
from msc_project.constants import STRESS_PREDICTION_PROJECT_NAME
from msc_project.scripts.utils import get_artifact_dataframe


def get_single_input_artifact(run):
    artifact = run.used_artifacts()
    assert len(artifact) == 1
    artifact = artifact[0]
    return artifact


def get_windowed_artifact(hrv_features_artifact):
    hrv_features_run = hrv_features_artifact.logged_by()
    heartpy_output_artifact = get_single_input_artifact(run=hrv_features_run)
    heartpy_output_run = heartpy_output_artifact.logged_by()
    windowed_artifact = get_single_input_artifact(run=heartpy_output_run)
    return windowed_artifact


if __name__ == "__main__":
    hrv_features_artifact_name: str = "hrv_features:v3"
    labels_artifact_name: str = "labels:v0"
    dataset_name: str = "raw_signal"
    config = {"dataset_name": dataset_name}

    run = wandb.init(
        project=STRESS_PREDICTION_PROJECT_NAME,
        job_type="get_dataset",
        save_code=True,
        config=config,
    )

    # load HRV features
    hrv_features = get_artifact_dataframe(
        run=run,
        artifact_or_name=hrv_features_artifact_name,
        pkl_filename=os.path.join("changed_label", f"{dataset_name}.pkl"),
    )

    # load labels
    labels = get_artifact_dataframe(
        run=run,
        artifact_or_name=labels_artifact_name,
        pkl_filename="labels.pkl",
    )

    # load noisy mask

    # filter examples by clean/noisy

    # upload to wandb
