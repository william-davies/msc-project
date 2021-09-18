import os

import wandb

from msc_project.constants import SheetNames, DENOISING_AUTOENCODER_PROJECT_NAME
from msc_project.scripts.evaluate_autoencoder import get_model
from msc_project.scripts.utils import get_artifact_dataframe


def get_signal_processed_data(run, data_split_artifact):
    train = get_artifact_dataframe(
        run=run,
        artifact_or_name=data_split_artifact,
        pkl_filename=os.path.join("intermediate_preprocessed", "train.pkl"),
    )
    val = get_artifact_dataframe(
        run=run,
        artifact_or_name=data_split_artifact,
        pkl_filename=os.path.join("intermediate_preprocessed", "val.pkl"),
    )
    noisy = get_artifact_dataframe(
        run=run,
        artifact_or_name=data_split_artifact,
        pkl_filename=os.path.join("intermediate_preprocessed", "noisy.pkl"),
    )
    return train, val, noisy


if __name__ == "__main__":
    sheet_name_to_evaluate_on = SheetNames.EMPATICA_LEFT_BVP.value
    data_split_version = 4
    model_artifact_name = "trained_on_EmLBVP:v0"

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type="model_evaluation",
        save_code=True,
        notes="comparison plot",
    )

    autoencoder = get_model(run=run, artifact_or_name=model_artifact_name)

    data_split_artifact_name = (
        f"{sheet_name_to_evaluate_on}_data_split:v{data_split_version}"
    )
    data_split_artifact = run.use_artifact(
        f"william-davies/{DENOISING_AUTOENCODER_PROJECT_NAME}/{data_split_artifact_name}"
    )

    (
        signal_processed_train,
        signal_processed_val,
        signal_processed_noisy,
    ) = get_signal_processed_data(run=run, data_split_artifact=data_split_artifact)
