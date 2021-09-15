import os

import wandb

from msc_project.constants import (
    DENOISING_AUTOENCODER_PROJECT_NAME,
    SheetNames,
    PREPROCESSED_DATA_ARTIFACT,
    BASE_DIR,
)
from msc_project.scripts.evaluate_autoencoder import get_model, get_reconstructed_df
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

    autoencoder = get_model(run=run, model_version=model_version)
    proposed_denoised_data = get_reconstructed_df(
        to_reconstruct=intermediate_preprocessed_data.T,
        autoencoder=autoencoder,
    ).T

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

    if upload_artifact:
        artifact = wandb.Artifact(name=f"{sheet_name}", type="preprocessed_data")
        artifact.add_dir(run_dir)
        run.log_artifact(artifact)
    run.finish()
