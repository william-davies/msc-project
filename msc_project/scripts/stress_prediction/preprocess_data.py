import wandb

from msc_project.constants import (
    DENOISING_AUTOENCODER_PROJECT_NAME,
    SheetNames,
    PREPROCESSED_DATA_ARTIFACT,
)
from msc_project.scripts.get_preprocessed_data import downsample, get_freq
from msc_project.scripts.hrv.get_hrv import get_artifact_dataframe

if __name__ == "__main__":
    run = wandb.init(
        project="stress-prediction",
        job_type="stress_prediction_preprocess_data",
        save_code=True,
    )
    sheet_name = SheetNames.INFINITY.value
    preprocessed_data_artifact_version: int = 2
    downsampled_rate: float = 16

    autoencoder_preprocessed_data_artifact = run.use_artifact(
        artifact_or_name=f"{DENOISING_AUTOENCODER_PROJECT_NAME}/{sheet_name}_preprocessed_data:v{preprocessed_data_artifact_version}",
        type=PREPROCESSED_DATA_ARTIFACT,
    )

    windowed_raw_data = get_artifact_dataframe(
        run=run,
        artifact_or_name=autoencoder_preprocessed_data_artifact,
        pkl_filename="inf_raw_data_hrv.pkl",
    )

    original_fs = get_freq(windowed_raw_data.index)
    downsampled_windowed_raw_data = downsample(
        original_data=windowed_raw_data,
        original_rate=original_fs,
        downsampled_rate=downsampled_rate,
    )
