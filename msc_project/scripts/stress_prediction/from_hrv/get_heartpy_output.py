"""
Get HRV features that will be used for stress prediction. I'm using a sliding window this time, so I made a different file
to `hrv/get_hrv`, where I didn't use a sliding window and I also had input signals from both Infiniti and Empatica.
"""
import os

import wandb

from msc_project.constants import (
    STRESS_PREDICTION_PROJECT_NAME,
    DENOISING_AUTOENCODER_PROJECT_NAME,
    PREPROCESSED_DATA_ARTIFACT,
)
from msc_project.scripts.hrv.get_hrv import get_hrv
from msc_project.scripts.hrv.get_whole_signal_hrv import add_temp_file_to_artifact
from msc_project.scripts.utils import get_artifact_dataframe


if __name__ == "__main__":
    windowed_data_artifact_name: str = "windowed:v0"
    upload_artifact: bool = True

    run = wandb.init(
        project=STRESS_PREDICTION_PROJECT_NAME,
        job_type="get_hrv_features",
        save_code=True,
    )

    # load raw, just downsampled, traditional preprocessed, DAE signal
    raw_signal = get_artifact_dataframe(
        run=run,
        artifact_or_name=windowed_data_artifact_name,
        pkl_filename="raw_signal.pkl",
    )
    just_downsampled_signal = get_artifact_dataframe(
        run=run,
        artifact_or_name=windowed_data_artifact_name,
        pkl_filename="just_downsampled_signal.pkl",
    )
    traditional_preprocessed_signal = get_artifact_dataframe(
        run=run,
        artifact_or_name=windowed_data_artifact_name,
        pkl_filename="traditional_preprocessed_signal.pkl",
    )
    dae_denoised_signal = get_artifact_dataframe(
        run=run,
        artifact_or_name=windowed_data_artifact_name,
        pkl_filename="dae_denoised_signal.pkl",
    )

    # compute HRV features
    print("raw_hrv")
    raw_hrv = get_hrv(signal_data=raw_signal)
    print("just_downsampled_hrv")
    just_downsampled_hrv = get_hrv(signal_data=just_downsampled_signal)
    print("traditional_preprocessed_hrv")
    traditional_preprocessed_hrv = get_hrv(signal_data=traditional_preprocessed_signal)
    print("dae_denoised_hrv")
    dae_denoised_hrv = get_hrv(signal_data=dae_denoised_signal)

    # save HRV features
    if upload_artifact:
        artifact = wandb.Artifact(name="hrv", type="get_hrv")
        add_temp_file_to_artifact(
            artifact=artifact, fp="raw_signal_hrv.pkl", df=raw_hrv
        )
        add_temp_file_to_artifact(
            artifact=artifact,
            fp="just_downsampled_signal_hrv.pkl",
            df=just_downsampled_hrv,
        )
        add_temp_file_to_artifact(
            artifact=artifact,
            fp="traditional_preprocessed_signal_hrv.pkl",
            df=traditional_preprocessed_hrv,
        )
        add_temp_file_to_artifact(
            artifact=artifact, fp="dae_denoised_signal_hrv.pkl", df=dae_denoised_hrv
        )
        run.log_artifact(artifact)
    run.finish()
