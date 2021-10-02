import os

import pandas as pd

import wandb
from msc_project.constants import STRESS_PREDICTION_PROJECT_NAME
from msc_project.scripts.utils import get_artifact_dataframe


def read_changed_label_df(artifact_or_name, pkl_filename, api):
    return get_artifact_dataframe(
        artifact_or_name=artifact_or_name,
        pkl_filename=os.path.join("changed_label", pkl_filename),
        api=api,
    )


def plot_bar_chat(hrv: pd.DataFrame):
    meaned = pd.DataFrame(
        index=hrv.index.get_level_values(level="treatment_label").unique(),
        columns=hrv.columns,
    )
    for treatment_label, df in hrv.groupby(axis=0, level="treatment_label"):
        meaned.loc[treatment_label] = df.mean(axis=0)


if __name__ == "__main__":
    api = wandb.Api()
    features_artifact_name: str = f"{STRESS_PREDICTION_PROJECT_NAME}/hrv_features:v3"

    raw_signal_hrv = read_changed_label_df(
        artifact_or_name=features_artifact_name, pkl_filename="raw_signal.pkl", api=api
    )
    just_downsampled_signal_hrv = read_changed_label_df(
        artifact_or_name=features_artifact_name,
        pkl_filename="just_downsampled_signal.pkl",
        api=api,
    )
    traditional_preprocessed_signal_hrv = read_changed_label_df(
        artifact_or_name=features_artifact_name,
        pkl_filename="traditional_preprocessed_signal.pkl",
        api=api,
    )
    dae_denoised_signal_hrv = read_changed_label_df(
        artifact_or_name=features_artifact_name,
        pkl_filename="dae_denoised_signal.pkl",
        api=api,
    )

    plot_bar_chat(hrv=raw_signal_hrv)
