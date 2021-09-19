import pandas as pd
import wandb

from msc_project.constants import SheetNames, DENOISING_AUTOENCODER_PROJECT_NAME
from msc_project.scripts.comparison_denoise_plot import (
    get_only_downsampled_data,
    get_raw_data,
)
from msc_project.scripts.evaluate_autoencoder import get_SQI

if __name__ == "__main__":
    sheet_name_to_evaluate_on = SheetNames.EMPATICA_LEFT_BVP.value
    data_split_version = 2

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type="model_evaluation",
        save_code=True,
        notes="compare sqi of raw and downsample",
    )

    data_split_artifact_name = (
        f"{sheet_name_to_evaluate_on}_data_split:v{data_split_version}"
    )
    data_split_artifact = run.use_artifact(
        f"william-davies/{DENOISING_AUTOENCODER_PROJECT_NAME}/{data_split_artifact_name}"
    )

    downsampled_train, downsampled_val, downsampled_noisy = get_only_downsampled_data(
        run=run, data_split_artifact=data_split_artifact
    )
    downsampled = pd.concat(
        objs=(downsampled_train, downsampled_val, downsampled_noisy)
    )

    raw_data = get_raw_data(
        data_split_artifact=data_split_artifact,
    )

    raw_sqi = get_SQI(data=raw_data)
    downsampled_sqi = get_SQI(data=downsampled)
