"""
Load HRV features (input) and labels (target). Filter by clean signals. Filter features to use.
Parameters
- hrv features to use
- participants to exclude
"""
import os
from typing import List, Dict

import pandas as pd
import wandb
from msc_project.constants import STRESS_PREDICTION_PROJECT_NAME
from msc_project.scripts.hrv.get_rmse import (
    get_clean_signal_indexes,
    metrics_of_interest,
)
from msc_project.scripts.hrv.utils import add_temp_file_to_artifact
from msc_project.scripts.stress_prediction.from_hrv.get_hrv_features import (
    change_treatment_labels,
    get_non_baseline_windows,
)
from msc_project.scripts.stress_prediction.from_hrv.get_preprocessed_data import (
    get_sheet_name_prefix,
)
from msc_project.scripts.stress_prediction.from_hrv.train_model import (
    preprocessing_methods,
)
from msc_project.scripts.utils import get_artifact_dataframe, get_single_input_artifact


def get_windowed_artifact(hrv_features_artifact):
    hrv_features_run = hrv_features_artifact.logged_by()
    heartpy_output_artifact = get_single_input_artifact(run=hrv_features_run)
    heartpy_output_run = heartpy_output_artifact.logged_by()
    windowed_artifact = get_single_input_artifact(run=heartpy_output_run)
    return windowed_artifact


def filter_labels(labels: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    I do different data filtering as I do experiments. Put them all in here so it's cleaner.
    :param labels:
    :return:
    """
    return labels.drop(index=config["excluded_participants"])


def filter_hrv_features(hrv_features: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    I do different data filtering as I do experiments. Put them all in here so it's cleaner.
    :param hrv_features:
    :return:
    """
    return hrv_features.drop(index=config["excluded_participants"]).loc[
        :, config["included_features"]
    ]


def get_combined_hrv_features(complete_dataset_artifact) -> pd.DataFrame:
    raw_hrv_features = get_artifact_dataframe(
        run=run,
        artifact_or_name=hrv_features_artifact_name,
        pkl_filename=os.path.join("changed_label", f"raw_signal.pkl"),
    )
    dae_denoised_hrv_features = get_artifact_dataframe(
        run=run,
        artifact_or_name=hrv_features_artifact_name,
        pkl_filename=os.path.join("changed_label", f"dae_denoised_signal.pkl"),
    )
    raw_features_to_combine = raw_hrv_features[["pnn50", "lf/hf"]]
    dae_denoised_features_to_combine = dae_denoised_hrv_features[
        ["bpm", "sdnn", "rmssd"]
    ]
    combined_hrv_features = pd.concat(
        objs=[raw_features_to_combine, dae_denoised_features_to_combine], axis=1
    )
    return combined_hrv_features


def assert_both_index_and_columns_are_equal(df1, df2):
    assert df1.index.equals(df2.index)
    assert df1.columns.equals(df2.columns)


if __name__ == "__main__":
    hrv_features_artifact_name: str = "EmLBVP_hrv_features:v0"
    labels_artifact_name: str = "EmLBVP_labels:v0"
    config = {
        "noise_tolerance": 1,
        "excluded_participants": ["0726114041P6_609"],
        "included_features": ["bpm", "sdnn", "rmssd", "pnn50", "lf/hf"],
    }
    notes = ""
    upload_to_wandb: bool = True

    assert get_sheet_name_prefix(hrv_features_artifact_name) == get_sheet_name_prefix(
        labels_artifact_name
    )
    sheet_name = get_sheet_name_prefix(hrv_features_artifact_name)

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
    filtered_labels = filter_labels(labels=labels, config=config)

    complete_dataset_artifact = wandb.Artifact(
        name=f"{sheet_name}_complete_dataset",
        type="get_dataset",
        metadata=config,
        description=notes,
    )

    # load HRV features
    for preprocessing_method in preprocessing_methods:
        hrv_features = get_artifact_dataframe(
            run=run,
            artifact_or_name=hrv_features_artifact_name,
            pkl_filename=os.path.join(
                "changed_label", f"{preprocessing_method}_signal.pkl"
            ),
        )

        filtered_hrv_features = filter_hrv_features(
            hrv_features=hrv_features, config=config
        )
        # check input and labels are in same order
        assert filtered_hrv_features.index.equals(filtered_labels.index)
        add_temp_file_to_artifact(
            artifact=complete_dataset_artifact,
            fp=f"{preprocessing_method}_signal_hrv_features.pkl",
            df=filtered_hrv_features,
        )

    # get combined hrv features
    combined_hrv_features = get_combined_hrv_features()
    combined_hrv_features = combined_hrv_features[filtered_hrv_features.columns]
    assert_both_index_and_columns_are_equal(combined_hrv_features, filtered_labels)

    # upload to wandb
    if upload_to_wandb:
        add_temp_file_to_artifact(
            artifact=complete_dataset_artifact,
            fp=f"combined_hrv_features.pkl",
            df=combined_hrv_features,
        )
        add_temp_file_to_artifact(
            artifact=complete_dataset_artifact,
            fp="stress_labels.pkl",
            df=filtered_labels,
        )
        complete_dataset_artifact.add_file(local_path=os.path.abspath(__file__))
        run.log_artifact(complete_dataset_artifact)
    run.finish()
