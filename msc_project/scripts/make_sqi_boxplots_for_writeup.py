"""
Compare original signal vs traditionally preprocessed signal vs DAE denoised signal.
"""
import os
from typing import Tuple

import numpy as np
import pandas as pd
import wandb
from matplotlib import pyplot as plt

from msc_project.constants import DENOISING_AUTOENCODER_PROJECT_NAME
from msc_project.scripts.compare_sqi import make_boxplots
from msc_project.scripts.evaluate_autoencoder import (
    download_artifact_model,
    download_preprocessed_data,
    get_SQI,
    SQI_HR_range_min,
    SQI_HR_range_max,
    get_reconstructed_df,
)
from msc_project.scripts.utils import get_committed_artifact_dataframe


def indexes_equal_ignore_order(index1, index2) -> bool:
    """
    pandas.Index.equals cares about order. This function doesn't care about order.
    :param index1:
    :param index2:
    :return:
    """
    is_equal_one_way = len(index1.difference(index2)) == 0
    is_equal_other_way = len(index2.difference(index1)) == 0
    return is_equal_one_way and is_equal_other_way


def make_boxplots_wrapper(sqis, signal_type: str) -> None:
    make_boxplots(all_sqis=sqis, labels=denoising_methods)
    plt.title(
        f"Comparing pSQI attained by traditional preprocessing\n and DAE denoising methods\n{signal_type} signals"
    )
    plt.tight_layout()
    plt.show()


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

    data_split_artifact = run.use_artifact(data_split_artifact_name)
    preprocessed_data_fp = download_preprocessed_data(data_split_artifact)

    # get all signals
    # transpose so examples is row axis.
    # get traditional preprocessed signals
    traditional_preprocessed_signals = pd.read_pickle(
        os.path.join(
            preprocessed_data_fp, "windowed", "traditional_preprocessed_data.pkl"
        )
    ).T

    # get original signal
    original_signals = pd.read_pickle(
        os.path.join(preprocessed_data_fp, "windowed", "raw_data.pkl")
    ).T
    original_signals = original_signals.loc[traditional_preprocessed_signals.index]

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
    clean_signals = pd.concat(objs=[train, val], axis=0)
    model_input_signals = pd.concat(objs=[clean_signals, noisy], axis=0)

    autoencoder = download_artifact_model(run=run, artifact_or_name=model_artifact_name)
    dae_denoised_signals = get_reconstructed_df(
        model_input_signals, autoencoder=autoencoder
    )

    # validation
    assert indexes_equal_ignore_order(
        traditional_preprocessed_signals.index, dae_denoised_signals.index
    )
    assert indexes_equal_ignore_order(
        original_signals.index, dae_denoised_signals.index
    )

    # get all SQIs
    (
        original_signals_SQI,
        traditional_preprocessed_signals_SQI,
        dae_denoised_signals_SQI,
    ) = (
        get_SQI(
            signal,
            band_of_interest_lower_freq=SQI_HR_range_min,
            band_of_interest_upper_freq=SQI_HR_range_max,
        )
        for signal in (
            original_signals,
            traditional_preprocessed_signals,
            dae_denoised_signals,
        )
    )

    # make plots
    all_sqis = (
        original_signals_SQI,
        traditional_preprocessed_signals_SQI,
        dae_denoised_signals_SQI,
    )
    all_sqis = tuple(map(pd.DataFrame.squeeze, all_sqis))

    denoising_methods = [
        "Original signal",
        "Traditional preprocessed signal",
        "DAE denoised signal",
    ]

    # clean signals
    clean_sqis = tuple(sqi.loc[clean_signals.index] for sqi in all_sqis)
    make_boxplots_wrapper(sqis=clean_sqis, signal_type="Clean")

    # noisy signals
    clean_sqis = tuple(sqi.loc[noisy.index] for sqi in all_sqis)
    make_boxplots_wrapper(sqis=clean_sqis, signal_type="Noisy")
