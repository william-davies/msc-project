"""
Compare original signal vs traditionally preprocessed signal vs DAE denoised signal.
"""
import os

import numpy as np
import pandas as pd
import wandb

from msc_project.constants import DENOISING_AUTOENCODER_PROJECT_NAME
from msc_project.scripts.evaluate_autoencoder import (
    download_artifact_model,
    download_preprocessed_data,
    get_SQI,
    SQI_HR_range_min,
    SQI_HR_range_max,
    get_reconstructed_df,
)
from msc_project.scripts.utils import get_committed_artifact_dataframe


def get_central_3_minutes(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Works on already windowed data.
    :param signals:
    :return:
    """
    window_starts = signals.index.get_level_values(level="window_start")
    central_3_minutes_index = np.logical_and(
        window_starts >= 60, window_starts <= 4 * 60
    )
    central_3_minutes_df = signals.iloc[central_3_minutes_index]
    return central_3_minutes_df


if __name__ == "__main__":
    data_split_artifact_name: str = "EmLBVP_data_split:v4"
    model_artifact_name: str = "trained_on_EmLBVP:v6"
    data_name: str = "only_downsampled"
    notes = ""

    config = {"data_name": data_name}

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type="model_evaluation",
        notes=notes,
        save_code=True,
        config=config,
    )

    # get all signals
    autoencoder = download_artifact_model(run=run, artifact_or_name=model_artifact_name)

    data_split_artifact = run.use_artifact(data_split_artifact_name)
    preprocessed_data_fp = download_preprocessed_data(data_split_artifact)

    # get original signal
    # transpose so examples is row axis. like train/val/noisy
    original_signals = pd.read_pickle(
        os.path.join(preprocessed_data_fp, "windowed", "raw_data.pkl")
    ).T
    original_signals = get_central_3_minutes(original_signals)

    # get traditional preprocessed signals
    traditional_preprocessed_data = pd.read_pickle(
        os.path.join(
            preprocessed_data_fp, "windowed", "traditional_preprocessed_data.pkl"
        )
    ).T

    # get DAE denoised signals
    train = get_committed_artifact_dataframe(
        run=run,
        artifact_or_name=data_split_artifact,
        pkl_filename=os.path.join(data_name, "train.pkl"),
    )
    val = get_committed_artifact_dataframe(
        run=run,
        artifact_or_name=data_split_artifact,
        pkl_filename=os.path.join(data_name, "val.pkl"),
    )
    noisy = get_committed_artifact_dataframe(
        run=run,
        artifact_or_name=data_split_artifact,
        pkl_filename=os.path.join(data_name, "noisy.pkl"),
    )
    model_input_signals = pd.concat(objs=[train, val, noisy], axis=0)

    reconstructed_signals = get_reconstructed_df(
        model_input_signals, autoencoder=autoencoder
    )

    # get all SQIs
    (original_signals_SQI, model_input_SQI, reconstructed_signals_SQI,) = (
        get_SQI(
            signal,
            band_of_interest_lower_freq=SQI_HR_range_min,
            band_of_interest_upper_freq=SQI_HR_range_max,
        )
        for signal in (
            original_signals,
            model_input_signals,
            reconstructed_signals,
        )
    )
