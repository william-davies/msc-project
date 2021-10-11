"""
Load HRV features (input) and labels (target). Do filtering: signal noisiness, participants, features.
Combine features to make best combination of preprocessing methods.
Parameters
- hrv features to use
- participants to exclude
"""
import os
from itertools import chain
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
    feature_sets,
)
from msc_project.scripts.utils import (
    get_committed_artifact_dataframe,
    get_single_input_artifact,
)
from msc_project.scripts.hrv.get_rmse import metrics_of_interest as all_hrv_features


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


def get_pending_artifact_dataframe(artifact, pkl_basename) -> pd.DataFrame:
    """
    For artifact that hasn't been COMMITTED yet.
    :param artifact:
    :param pkl_basename:
    :return:
    """
    local_path = artifact.manifest.entries[pkl_basename].local_path
    return pd.read_pickle(local_path)


def get_combined_hrv_features(dataset_dataframes: Dict) -> pd.DataFrame:
    """
    Use hrv features mix-and-match from different feature sets to try to push up performance.
    :param dataset_dataframes:
    :return:
    """
    to_combine = []
    for feature_set, features_to_combine in combination_feature_set_config.items():
        features = dataset_dataframes[f"{feature_set}_signal_hrv_features"]
        to_combine.append(features[features_to_combine])
    combined_hrv_features = pd.concat(objs=to_combine, axis=1)
    mapper = lambda method: list(all_hrv_features).index(method)
    key = lambda index: index.map(mapper=mapper)
    sorted = combined_hrv_features.sort_index(axis=1, key=key)
    return sorted


def assert_both_index_and_columns_are_equal(df1, df2):
    assert df1.index.equals(df2.index)
    assert df1.columns.equals(df2.columns)


def validate_dataset(hrv_features_dict: Dict, labels: pd.DataFrame) -> None:
    """
    Check examples are in the same order. Check features are in the same order.
    :param hrv_features_dict:
    :return:
    """
    raw_hrv_features = hrv_features_dict["raw_signal_hrv_features"]
    assert raw_hrv_features.index.equals(labels.index)
    for feature_set_name in [
        "just_downsampled_signal",
        "traditional_preprocessed_signal",
        "dae_denoised_signal",
        "combined",
    ]:
        key = f"{feature_set_name}_hrv_features"
        feature_set_df = hrv_features_dict[key]
        assert_both_index_and_columns_are_equal(raw_hrv_features, feature_set_df)


def load_hrv_features(hrv_features_artifact_name: str):
    """
    Load HRV features. Filter HRV features. Add to artifact. Track in dictionary.
    :param hrv_features_artifact_name:
    :return:
    """
    sheet_name = get_sheet_name_prefix(hrv_features_artifact_name)
    hrv_features_dict = {}
    for feature_set in feature_sets[:-1]:
        hrv_features = get_committed_artifact_dataframe(
            run=run,
            artifact_or_name=hrv_features_artifact_name,
            pkl_filename=os.path.join("changed_label", f"{feature_set}.pkl"),
        )

        filtered_hrv_features = filter_hrv_features(
            hrv_features=hrv_features, config=config
        )
        add_temp_file_to_artifact(
            artifact=complete_dataset_artifact,
            fp=os.path.join(sheet_name, f"{feature_set}_hrv_features.pkl"),
            df=filtered_hrv_features,
        )
        hrv_features_dict[f"{feature_set}_hrv_features"] = filtered_hrv_features

    combined_hrv_features = get_combined_hrv_features(hrv_features_dict)
    add_temp_file_to_artifact(
        artifact=complete_dataset_artifact,
        fp=os.path.join(sheet_name, "combined_hrv_features.pkl"),
        df=combined_hrv_features,
    )
    hrv_features_dict["combined_hrv_features"] = combined_hrv_features
    return hrv_features_dict


def load_labels() -> pd.DataFrame:
    labels = get_committed_artifact_dataframe(
        run=run,
        artifact_or_name=labels_artifact_name,
        pkl_filename="labels.pkl",
    )
    filtered_labels = filter_labels(labels=labels, config=config)
    add_temp_file_to_artifact(
        artifact=complete_dataset_artifact,
        fp="stress_labels.pkl",
        df=filtered_labels,
    )
    return filtered_labels


def get_included_features(combination_feature_set_config):
    """
    Neat because we don't have to set included_features in two places.
    :param combination_feature_set_config:
    :return:
    """
    included_features = [
        features for features in combination_feature_set_config.values()
    ]
    return list(chain.from_iterable(included_features))


if __name__ == "__main__":
    Inf_features_artifact_name: str = "Inf_hrv_features:v0"
    EmLBVP_features_artifact_name: str = "EmLBVP_hrv_features:v0"
    labels_artifact_name: str = "labels:v0"
    Inf_excluded_participants = ["0725135216P4_608", "0726094551P5_609"]
    EmLBVP_excluded_participants = ["0726114041P6_609"]
    combined_excluded_participants = (
        Inf_excluded_participants + EmLBVP_excluded_participants
    )
    combination_feature_set_config = {
        "raw": ["pnn50", "lf", "hf", "lf/hf"],
        "dae_denoised": ["bpm", "sdnn", "rmssd"],
    }
    included_features = get_included_features(combination_feature_set_config)
    config = {
        "noise_tolerance": 1,
        "excluded_participants": combined_excluded_participants,
        "included_features": included_features,
        "combination_feature_set_config": combination_feature_set_config,
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

    complete_dataset_artifact = wandb.Artifact(
        name="complete_dataset",
        type="get_dataset",
        metadata=config,
        description=notes,
    )

    stress_labels = load_labels()

    for features_artifact_name in [
        Inf_features_artifact_name,
        EmLBVP_features_artifact_name,
    ]:
        sheet_name = get_sheet_name_prefix(features_artifact_name)
        hrv_features_dict = load_hrv_features(
            hrv_features_artifact_name=features_artifact_name
        )
        validate_dataset(hrv_features_dict=hrv_features_dict, labels=stress_labels)

    # upload to wandb
    if upload_to_wandb:
        complete_dataset_artifact.add_file(local_path=os.path.abspath(__file__))
        run.log_artifact(complete_dataset_artifact)
    run.finish()
