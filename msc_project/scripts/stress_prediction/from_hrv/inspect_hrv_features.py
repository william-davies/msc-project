import os

import pandas as pd
from matplotlib import pyplot as plt

import wandb
from msc_project.constants import STRESS_PREDICTION_PROJECT_NAME, BASE_DIR
from msc_project.scripts.utils import get_artifact_dataframe, slugify


def read_changed_label_df(artifact_or_name, pkl_filename, api):
    return get_artifact_dataframe(
        artifact_or_name=artifact_or_name,
        pkl_filename=os.path.join("changed_label", pkl_filename),
        api=api,
    )


def plot_bar_charts(hrv: pd.DataFrame, dataset_name: str):
    meaned = pd.DataFrame(
        index=hrv.index.get_level_values(level="treatment_label").unique(),
        columns=hrv.columns,
    )
    for treatment_label, df in hrv.groupby(axis=0, level="treatment_label"):
        meaned.loc[treatment_label] = df.mean(axis=0)

    dirpath = os.path.join(BASE_DIR, "results", "inspect_hrv_features", dataset_name)
    os.makedirs(dirpath, exist_ok=True)
    plt.figure()
    for feature in meaned.columns:
        plt.title(dataset_name)
        plt.ylabel(feature)
        means = meaned[feature]
        means.plot.bar()
        plt.tight_layout()
        plt.savefig(os.path.join(dirpath, slugify(feature)))
        plt.clf()


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

    plot_bar_charts(hrv=raw_signal_hrv, dataset_name="raw")
    plot_bar_charts(hrv=just_downsampled_signal_hrv, dataset_name="just downsampled")
    plot_bar_charts(
        hrv=traditional_preprocessed_signal_hrv, dataset_name="traditional preprocessed"
    )
    plot_bar_charts(hrv=dae_denoised_signal_hrv, dataset_name="dae denoised")
