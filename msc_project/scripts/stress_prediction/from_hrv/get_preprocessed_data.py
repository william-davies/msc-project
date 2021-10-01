"""
Perform sliding window on signals. Save noisy mask.
"""
import os

import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import wandb

from msc_project.constants import (
    STRESS_PREDICTION_PROJECT_NAME,
    DENOISING_AUTOENCODER_PROJECT_NAME,
    PREPROCESSED_DATA_ARTIFACT,
    SECONDS_IN_MINUTE,
)
from msc_project.scripts.data_processing.get_preprocessed_data import (
    get_freq,
    handle_data_windowing,
    get_temporal_subwindow_of_signal,
)
from msc_project.scripts.hrv.get_hrv import get_hrv
from msc_project.scripts.hrv.get_whole_signal_hrv import add_temp_file_to_artifact
from msc_project.scripts.utils import get_artifact_dataframe, safe_float_to_int


def get_preprocessed_data_artifact(dae_denoised_artifact):
    """
    Get preprocessed data artifact that contains the signal that was denoised to create the `dae_denoised_artifact`.
    :param dae_denoised_artifact:
    :return:
    """
    run = dae_denoised_artifact.logged_by()
    artifacts = run.used_artifacts()
    preprocessed_data_artifact = [
        artifact
        for artifact in artifacts
        if artifact.type == PREPROCESSED_DATA_ARTIFACT
    ]
    assert len(preprocessed_data_artifact) == 1
    preprocessed_data_artifact = preprocessed_data_artifact[0]
    return preprocessed_data_artifact


if __name__ == "__main__":
    dae_denoised_data_artifact_name: str = (
        f"{DENOISING_AUTOENCODER_PROJECT_NAME}/Inf_merged_signal:v0"
    )
    upload_artifact: bool = True

    config = {
        "window_duration": 2 * SECONDS_IN_MINUTE,
        "step_duration": 1 * SECONDS_IN_MINUTE,
    }

    run = wandb.init(
        project=STRESS_PREDICTION_PROJECT_NAME,
        job_type="get_hrv_features",
        save_code=True,
        config=config,
    )

    # get artifacts
    dae_denoised_artifact = run.use_artifact(dae_denoised_data_artifact_name)
    preprocessed_data_artifact = get_preprocessed_data_artifact(dae_denoised_artifact)

    # load Infiniti raw, just downsampled, traditional preprocessed signal, dae denoised
    raw_signal = get_artifact_dataframe(
        run=run,
        artifact_or_name=preprocessed_data_artifact,
        pkl_filename=os.path.join("not_windowed", "raw_data.pkl"),
    )
    raw_signal = get_temporal_subwindow_of_signal(
        df=raw_signal,
        window_start=SECONDS_IN_MINUTE,
        window_end=4 * SECONDS_IN_MINUTE,
    )

    just_downsampled_signal = get_artifact_dataframe(
        run=run,
        artifact_or_name=preprocessed_data_artifact,
        pkl_filename=os.path.join("not_windowed", "only_downsampled_data.pkl"),
    )
    traditional_preprocessed_signal = get_artifact_dataframe(
        run=run,
        artifact_or_name=preprocessed_data_artifact,
        pkl_filename=os.path.join("not_windowed", "traditional_preprocessed_data.pkl"),
    )
    dae_denoised_signal = get_artifact_dataframe(
        run=run,
        artifact_or_name=dae_denoised_artifact,
        pkl_filename="merged_signal.pkl",
    )
    noisy_mask = get_artifact_dataframe(
        run=run,
        artifact_or_name=preprocessed_data_artifact,
        pkl_filename=os.path.join("not_windowed", "noisy_mask.pkl"),
    )

    # sliding window
    raw_windowed = handle_data_windowing(
        non_windowed_data=raw_signal,
        window_duration=config["window_duration"],
        step_duration=config["step_duration"],
    )
    just_downsampled_windowed = handle_data_windowing(
        non_windowed_data=just_downsampled_signal,
        window_duration=config["window_duration"],
        step_duration=config["step_duration"],
    )
    traditional_preprocessed_windowed = handle_data_windowing(
        non_windowed_data=traditional_preprocessed_signal,
        window_duration=config["window_duration"],
        step_duration=config["step_duration"],
    )
    dae_denoised_windowed = handle_data_windowing(
        non_windowed_data=dae_denoised_signal,
        window_duration=config["window_duration"],
        step_duration=config["step_duration"],
    )
    noisy_mask_windowed = handle_data_windowing(
        non_windowed_data=noisy_mask,
        window_duration=config["window_duration"],
        step_duration=config["step_duration"],
    )

    # save HRV features
    if upload_artifact:
        artifact = wandb.Artifact(
            name="windowed", type="preprocessed_data", metadata=config
        )
        add_temp_file_to_artifact(
            artifact=artifact, fp="raw_signal.pkl", df=raw_windowed
        )
        add_temp_file_to_artifact(
            artifact=artifact,
            fp="just_downsampled_signal.pkl",
            df=just_downsampled_windowed,
        )
        add_temp_file_to_artifact(
            artifact=artifact,
            fp="traditional_preprocessed_signal.pkl",
            df=traditional_preprocessed_windowed,
        )
        add_temp_file_to_artifact(
            artifact=artifact, fp="dae_denoised_signal.pkl", df=dae_denoised_windowed
        )
        add_temp_file_to_artifact(
            artifact=artifact, fp="noisy_mask.pkl", df=noisy_mask_windowed
        )
        run.log_artifact(artifact)
    run.finish()
