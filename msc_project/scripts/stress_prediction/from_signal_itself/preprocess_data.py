"""
Preprocess data for stress prediction task.
1. Downsample raw data
2. Get proposed denoised data
3. Copy over traditional and intermediate preprocessed data
"""
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import wandb

from msc_project.constants import (
    DENOISING_AUTOENCODER_PROJECT_NAME,
    SheetNames,
    PREPROCESSED_DATA_ARTIFACT,
    BASE_DIR,
    STRESS_PREDICTION_PROJECT_NAME,
)
from msc_project.scripts.evaluate_autoencoder import (
    download_artifact_model,
    get_reconstructed_df,
)
from msc_project.scripts.get_preprocessed_data import downsample, get_freq
from msc_project.scripts.utils import get_artifact_dataframe


def get_labels(windowed_data: pd.DataFrame) -> pd.DataFrame:
    """

    :param windowed_data:
    :return:
    """
    label_df = pd.DataFrame(
        data=False, columns=windowed_data.columns, index=["is_high_stress"]
    )
    label_df.loc["is_high_stress", (slice(None), "m2_hard")] = True
    label_df.loc["is_high_stress", (slice(None), "m4_hard")] = True
    return label_df


def plot_window(window_index):
    plt.plot(raw_data[window_index], label="original")
    plt.plot(only_downsampled_data[window_index], label="downsampled")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run = wandb.init(
        project=STRESS_PREDICTION_PROJECT_NAME,
        job_type="preprocess_data",
        save_code=True,
    )
    sheet_name = SheetNames.INFINITY.value
    preprocessed_data_artifact_version: int = 2
    downsampled_rate: float = 16
    model_version: int = 40
    upload_artifact: bool = False

    run_dir = os.path.join(BASE_DIR, "data", "stress_prediction", sheet_name, run.name)
    os.makedirs(run_dir)

    autoencoder_preprocessed_data_artifact = run.use_artifact(
        artifact_or_name=f"{DENOISING_AUTOENCODER_PROJECT_NAME}/{sheet_name}_preprocessed_data:v{preprocessed_data_artifact_version}",
        type=PREPROCESSED_DATA_ARTIFACT,
    )

    raw_data = get_artifact_dataframe(
        run=run,
        artifact_or_name=autoencoder_preprocessed_data_artifact,
        pkl_filename="windowed_raw_data.pkl",
    )
    traditional_preprocessed_data = get_artifact_dataframe(
        run=run,
        artifact_or_name=autoencoder_preprocessed_data_artifact,
        pkl_filename="windowed_traditional_preprocessed_data.pkl",
    )
    intermediate_preprocessed_data = get_artifact_dataframe(
        run=run,
        artifact_or_name=autoencoder_preprocessed_data_artifact,
        pkl_filename="windowed_intermediate_preprocessed_data.pkl",
    )

    original_fs = get_freq(raw_data.index)
    only_downsampled_data = downsample(
        original_data=raw_data,
        original_rate=original_fs,
        downsampled_rate=downsampled_rate,
    )

    autoencoder = download_artifact_model(
        run=run, model_version=model_version, project=DENOISING_AUTOENCODER_PROJECT_NAME
    )
    proposed_denoised_data = get_reconstructed_df(
        to_reconstruct=intermediate_preprocessed_data.T,
        autoencoder=autoencoder,
    ).T

    labels = get_labels(windowed_data=raw_data)

    only_downsampled_data.to_pickle(os.path.join(run_dir, "only_downsampled_data.pkl"))
    traditional_preprocessed_data.to_pickle(
        os.path.join(run_dir, "traditional_preprocessed_data.pkl")
    )
    intermediate_preprocessed_data.to_pickle(
        os.path.join(run_dir, "intermediate_preprocessed_data.pkl")
    )
    proposed_denoised_data.to_pickle(
        os.path.join(run_dir, "proposed_denoised_data.pkl")
    )
    labels.to_pickle(os.path.join(run_dir, "labels.pkl"))

    if upload_artifact:
        artifact = wandb.Artifact(name=f"{sheet_name}", type="preprocessed_data")
        artifact.add_dir(run_dir)
        run.log_artifact(artifact)
    run.finish()

    num_examples_to_plot: int = 20
    window_indexes = np.random.choice(
        a=raw_data.columns, size=num_examples_to_plot, replace=False
    )
    for window_index in window_indexes:
        plot_window(window_index)
