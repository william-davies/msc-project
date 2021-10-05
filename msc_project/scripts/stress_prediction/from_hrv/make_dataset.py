"""
Load HRV features (input) and labels (target). Filter by clean signals.
"""
import os
from typing import List

import pandas as pd
import wandb
from msc_project.constants import STRESS_PREDICTION_PROJECT_NAME
from msc_project.scripts.hrv.get_rmse import get_clean_signal_indexes
from msc_project.scripts.hrv.get_whole_signal_hrv import add_temp_file_to_artifact
from msc_project.scripts.stress_prediction.from_hrv.get_hrv_features import (
    change_treatment_labels,
    get_non_baseline_windows,
)
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


def filter_data(data, excluded_participants: List[str]) -> pd.DataFrame:
    """
    I do different data filtering as I do experiments. Put them all in here so it's cleaner.
    :param data:
    :return:
    """
    return data.drop(index=excluded_participants)


if __name__ == "__main__":
    hrv_features_artifact_name: str = "hrv_features:v4"
    labels_artifact_name: str = "labels:v0"
    dataset_names: List[str] = [
        "raw",
        "just_downsampled",
        "traditional_preprocessed",
        "dae_denoised",
    ]
    config = {
        "noise_tolerance": 1,
        "excluded_participants": ["0725135216P4_608", "0726094551P5_609"],
    }
    notes = ""
    upload_to_wandb: bool = True

    run = wandb.init(
        project=STRESS_PREDICTION_PROJECT_NAME,
        job_type="get_dataset",
        save_code=True,
        config=config,
        notes=notes,
    )

    # get clean indexes
    windowed_artifact = get_windowed_artifact(
        hrv_features_artifact=run.use_artifact(
            artifact_or_name=hrv_features_artifact_name
        )
    )
    noisy_mask = get_artifact_dataframe(
        run=run,
        artifact_or_name=windowed_artifact,
        pkl_filename="noisy_mask.pkl",
    )
    noisy_mask = change_treatment_labels(noisy_mask.T).T
    noisy_proportions = noisy_mask.sum(axis=0) / noisy_mask.shape[0]
    clean_indexes = get_clean_signal_indexes(
        noisy_mask=noisy_mask, noise_tolerance=config["noise_tolerance"]
    )
    clean_indexes = get_non_baseline_windows(clean_indexes)

    # load labels
    labels = get_artifact_dataframe(
        run=run,
        artifact_or_name=labels_artifact_name,
        pkl_filename="labels.pkl",
    )
    filtered_labels = filter_data(
        data=labels, excluded_participants=config["excluded_participants"]
    )

    artifact = wandb.Artifact(
        name="complete_dataset",
        type="get_dataset",
        metadata=config,
        description=notes,
    )

    # load HRV features
    for dataset_name in dataset_names:
        hrv_features = get_artifact_dataframe(
            run=run,
            artifact_or_name=hrv_features_artifact_name,
            pkl_filename=os.path.join("changed_label", f"{dataset_name}_signal.pkl"),
        )

        filtered_hrv_features = filter_data(
            data=hrv_features, excluded_participants=config["excluded_participants"]
        )
        # check input and labels are in same order
        assert filtered_hrv_features.index.equals(filtered_labels.index)
        add_temp_file_to_artifact(
            artifact=artifact,
            fp=f"{dataset_name}_signal_hrv_features.pkl",
            df=filtered_hrv_features,
        )

    # upload to wandb
    if upload_to_wandb:
        add_temp_file_to_artifact(
            artifact=artifact, fp="stress_labels.pkl", df=filtered_labels
        )
        run.log_artifact(artifact)
    run.finish()
