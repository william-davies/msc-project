"""
Get HRV of full 3 minute signals.
"""
import os
from typing import List

import pandas as pd
import wandb

from msc_project.constants import (
    DENOISING_AUTOENCODER_PROJECT_NAME,
    SECONDS_IN_MINUTE,
    BASE_DIR,
)
from msc_project.scripts.get_preprocessed_data import get_temporal_subwindow_of_signal
from msc_project.scripts.hrv.get_hrv import get_hrv
from msc_project.scripts.hrv.inspect_heartpy_output import heartpy_plotter_wrapper
from msc_project.scripts.utils import get_artifact_dataframe, slugify
import warnings


def plot_heartpy_outputs(heartpy_output, dataset_name: str):
    save_dir = os.path.join(plots_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    heartpy_successful = heartpy_output.columns[~heartpy_output.isna().all()]
    for treatment in heartpy_successful:
        heartpy_plotter_wrapper(
            working_data=heartpy_output[treatment],
            measures=heartpy_output[treatment],
            title=f"{dataset_name}-{treatment}",
            show=False,
            save_filepath=os.path.join(save_dir, slugify(treatment)),
            figsize=(18, 9),
        )


def get_hrv_wrapper(signal_data):
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings("always")
        hrv = get_hrv(signal_data=signal_data)
        return hrv, w


def get_warning_msgs(warnings) -> List[str]:
    """
    Get just the messages so I can quickly debug.
    :param warnings:
    :return:
    """
    return [warning.message.args[0] for warning in warnings]


def add_temp_file_to_artifact(artifact, fp: str, df: pd.DataFrame) -> None:
    """
    Helper function
    :param artifact:
    :param fp:
    :param df:
    :return:
    """
    with artifact.new_file(fp) as f:
        df.to_pickle(f.name)


if __name__ == "__main__":
    infinity_preprocessed_data_artifact_name = "Inf_preprocessed_data:v6"
    empatica_preprocessed_data_artifact_name = "EmLBVP_preprocessed_data:v6"
    dae_denoised_signal_artifact_name = "EmLBVP_merged_signal:v1"
    make_plots: bool = False
    upload_artifact: bool = True

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type="get_hrv",
        save_code=True,
    )

    # load raw infinity gt signal
    inf_raw = get_artifact_dataframe(
        run=run,
        artifact_or_name=infinity_preprocessed_data_artifact_name,
        pkl_filename=os.path.join("not_windowed", "raw_data.pkl"),
    )
    #   crop central 3 minutes
    inf_raw = get_temporal_subwindow_of_signal(
        df=inf_raw,
        window_start=SECONDS_IN_MINUTE,
        window_end=4 * SECONDS_IN_MINUTE,
    )

    # load just downsampled empatica signal
    empatica_only_downsampled = get_artifact_dataframe(
        run=run,
        artifact_or_name=empatica_preprocessed_data_artifact_name,
        pkl_filename=os.path.join("not_windowed", "only_downsampled_data.pkl"),
    )

    # load traditional preprocessed signal
    empatica_traditional_preprocessed = get_artifact_dataframe(
        run=run,
        artifact_or_name=empatica_preprocessed_data_artifact_name,
        pkl_filename=os.path.join("not_windowed", "traditional_preprocessed_data.pkl"),
    )

    # load DAE denoised signal
    empatica_dae_denoised = get_artifact_dataframe(
        run=run,
        artifact_or_name=dae_denoised_signal_artifact_name,
        pkl_filename="merged_signal.pkl",
    )

    # get HRV of all treatments
    inf_raw_hrv, inf_raw_warnings = get_hrv_wrapper(signal_data=inf_raw)
    empatica_only_downsampled_hrv, empatica_only_downsampled_warnings = get_hrv_wrapper(
        signal_data=empatica_only_downsampled
    )
    (
        empatica_traditional_preprocessed_hrv,
        empatica_traditional_preprocessed_warnings,
    ) = get_hrv_wrapper(signal_data=empatica_traditional_preprocessed)
    empatica_dae_denoised_hrv, empatica_dae_denoised_warnings = get_hrv_wrapper(
        signal_data=empatica_dae_denoised
    )

    warning_msgs = get_warning_msgs(warnings=empatica_dae_denoised_warnings)

    if make_plots:
        plots_dir = os.path.join(BASE_DIR, "results", "hrv", run.name)
        plot_heartpy_outputs(heartpy_output=inf_raw_hrv, dataset_name="inf_raw")
        plot_heartpy_outputs(
            heartpy_output=empatica_only_downsampled_hrv,
            dataset_name="empatica_only_downsampled",
        )
        plot_heartpy_outputs(
            heartpy_output=empatica_traditional_preprocessed_hrv,
            dataset_name="empatica_traditional_preprocessed",
        )
        plot_heartpy_outputs(
            heartpy_output=empatica_dae_denoised_hrv,
            dataset_name="empatica_dae_denoised",
        )

    # upload HRV of all treatments
    if upload_artifact:
        hrv_artifact = wandb.Artifact(name="get_merged_signal_hrv", type="get_hrv")
        add_temp_file_to_artifact(
            artifact=hrv_artifact, fp="inf_raw.pkl", df=inf_raw_hrv
        )
        add_temp_file_to_artifact(
            artifact=hrv_artifact,
            fp="empatica_only_downsampled.pkl",
            df=empatica_only_downsampled_hrv,
        )
        add_temp_file_to_artifact(
            artifact=hrv_artifact,
            fp="empatica_traditional_preprocessed.pkl",
            df=empatica_traditional_preprocessed_hrv,
        )
        add_temp_file_to_artifact(
            artifact=hrv_artifact,
            fp="empatica_dae_denoised.pkl",
            df=empatica_dae_denoised_hrv,
        )
        run.log_artifact(hrv_artifact)
    run.finish()
