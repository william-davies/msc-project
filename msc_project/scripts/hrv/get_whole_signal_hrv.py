"""
Get HRV of full 3 minute signals.
"""
import os

import wandb

from msc_project.constants import DENOISING_AUTOENCODER_PROJECT_NAME, SECONDS_IN_MINUTE
from msc_project.scripts.get_preprocessed_data import get_temporal_subwindow_of_signal
from msc_project.scripts.hrv.get_hrv import get_hrv
from msc_project.scripts.utils import get_artifact_dataframe
import warnings

if __name__ == "__main__":

    infinity_preprocessed_data_artifact_name = "Inf_preprocessed_data:v6"
    empatica_preprocessed_data_artifact_name = "EmLBVP_preprocessed_data:v6"
    dae_denoised_signal_artifact_name = "EmLBVP_merged_signal:v1"

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
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings("always")
        inf_raw_hrv = get_hrv(signal_data=inf_raw)
    # empatica_only_downsampled_hrv = get_hrv(signal_data=empatica_only_downsampled)
    # empatica_traditional_preprocessed_hrv = get_hrv(signal_data=empatica_traditional_preprocessed)
    # empatica_dae_denoised_hrv = get_hrv(signal_data=empatica_dae_denoised)

    # upload HRV of all treatments
