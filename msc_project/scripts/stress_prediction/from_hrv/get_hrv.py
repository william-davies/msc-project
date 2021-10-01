"""
Get HRV features that will be used for stress prediction. I'm using a sliding window this time, so I made a different file
to `hrv/get_hrv`, where I didn't use a sliding window and I also had input signals from both Infiniti and Empatica.
"""
import wandb

from msc_project.constants import (
    STRESS_PREDICTION_PROJECT_NAME,
    DENOISING_AUTOENCODER_PROJECT_NAME,
    PREPROCESSED_DATA_ARTIFACT,
)
from msc_project.scripts.utils import get_artifact_dataframe


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

    run = wandb.init(
        project=STRESS_PREDICTION_PROJECT_NAME,
        job_type="get_hrv_features",
        save_code=True,
    )

    dae_denoised_artifact = run.use_artifact(dae_denoised_data_artifact_name)

    # load Infiniti raw data, just downsampled, traditional preprocessing, DAE denoised
    inf_raw_data = get_artifact_dataframe(
        run=run,
        artifact_or_name="Inf_preprocessed_data:v4",
        pkl_filename="windowed_raw_data.pkl",
    )

    # sliding window

    # compute HRV features

    # save HRV features
