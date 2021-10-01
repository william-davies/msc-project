"""
Denoise signal using DAE. Upload to wandb.
"""
import os

import wandb

from msc_project.constants import DENOISING_AUTOENCODER_PROJECT_NAME
from msc_project.scripts.utils import get_artifact_dataframe

if __name__ == "__main__":
    preprocessed_data_artifact_name: str = "Inf_preprocessed_data:v7"

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type="dae_denoise_signal",
        save_code=True,
    )

    # load just downsampled signal
    just_downsampled = get_artifact_dataframe(
        run=run,
        artifact_or_name=preprocessed_data_artifact_name,
        pkl_filename=os.path.join("not_windowed", "windowed_raw_data.pkl"),
    )
    # load model

    # dae denoise signal

    # upload to wandb
