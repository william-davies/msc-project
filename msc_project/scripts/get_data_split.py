"""
Split data into train, val, noisy. Don't do any other processing of data itself.
"""
import os

import wandb

from msc_project.constants import (
    DENOISING_AUTOENCODER_PROJECT_NAME,
    PREPROCESSED_DATA_ARTIFACT,
    BASE_DIR,
    DATA_SPLIT_ARTIFACT,
    SheetNames,
)
from msc_project.scripts.utils import get_artifact_dataframe
from msc_project.scripts.stress_prediction.from_signal_itself.get_data_split import (
    handle_data_split,
)


if __name__ == "__main__":
    sheet_name = SheetNames.INFINITY.value
    preprocessed_data_artifact_version: int = 8
    config = {"noise_tolerance": 0}
    job_type = "data_split"
    upload_artifact: bool = False

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type=job_type,
        config=config,
        save_code=True,
    )

    run_dir = os.path.join(
        BASE_DIR,
        "data",
        DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type,
        sheet_name,
        run.name,
    )
    os.makedirs(run_dir)

    preprocessed_data_artifact = run.use_artifact(
        artifact_or_name=f"{sheet_name}_preprocessed_data:v{preprocessed_data_artifact_version}",
        type=PREPROCESSED_DATA_ARTIFACT,
    )

    only_downsampled_data = get_artifact_dataframe(
        run=run,
        artifact_or_name=preprocessed_data_artifact,
        pkl_filename=os.path.join("windowed", "only_downsampled_data.pkl"),
    )
    traditional_preprocessed_data = get_artifact_dataframe(
        run=run,
        artifact_or_name=preprocessed_data_artifact,
        pkl_filename=os.path.join("windowed", "traditional_preprocessed_data.pkl"),
    )
    intermediate_preprocessed_data = get_artifact_dataframe(
        run=run,
        artifact_or_name=preprocessed_data_artifact,
        pkl_filename=os.path.join("windowed", "intermediate_preprocessed_data.pkl"),
    )
    noisy_mask = get_artifact_dataframe(
        run=run,
        artifact_or_name=preprocessed_data_artifact,
        pkl_filename=os.path.join("windowed", "noisy_mask.pkl"),
    )

    handle_data_split(
        signals=only_downsampled_data,
        save_dir=os.path.join(run_dir, "only_downsampled"),
        noise_tolerance=config["noise_tolerance"],
        noisy_mask=noisy_mask,
    )
    handle_data_split(
        signals=traditional_preprocessed_data,
        save_dir=os.path.join(run_dir, "traditional_preprocessed"),
        noise_tolerance=config["noise_tolerance"],
        noisy_mask=noisy_mask,
    )
    handle_data_split(
        signals=intermediate_preprocessed_data,
        save_dir=os.path.join(run_dir, "intermediate_preprocessed"),
        noise_tolerance=config["noise_tolerance"],
        noisy_mask=noisy_mask,
    )

    if upload_artifact:
        artifact = wandb.Artifact(
            name=f"{sheet_name}_{job_type}", type=job_type, metadata=config
        )
        artifact.add_dir(run_dir)
        run.log_artifact(artifact, type=job_type)
    run.finish()
