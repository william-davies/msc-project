import os

import pandas as pd
import wandb

from msc_project.constants import SheetNames, DENOISING_AUTOENCODER_PROJECT_NAME
from msc_project.scripts.evaluate_autoencoder import (
    get_model,
    download_preprocessed_data,
)
from msc_project.scripts.utils import get_artifact_dataframe


def get_signal_processed_data(run, data_split_artifact):
    return get_data_split(
        run=run,
        data_split_artifact=data_split_artifact,
        data_name="intermediate_preprocessed",
    )


def get_data_split(run, data_split_artifact, data_name):
    """

    :param run:
    :param data_split_artifact:
    :param data_name: only_downsampled/intermediate_preprocessed
    :return:
    """
    train = get_artifact_dataframe(
        run=run,
        artifact_or_name=data_split_artifact,
        pkl_filename=os.path.join(data_name, "train.pkl"),
    )
    val = get_artifact_dataframe(
        run=run,
        artifact_or_name=data_split_artifact,
        pkl_filename=os.path.join(data_name, "val.pkl"),
    )
    noisy = get_artifact_dataframe(
        run=run,
        artifact_or_name=data_split_artifact,
        pkl_filename=os.path.join(data_name, "noisy.pkl"),
    )
    return train, val, noisy


def get_raw_data(data_split_artifact, train_index, val_index, noisy_index):
    preprocessed_data_fp = download_preprocessed_data(data_split_artifact)
    # transpose so examples is row axis. like train/val/noisy
    raw_data = pd.read_pickle(
        os.path.join(preprocessed_data_fp, "windowed_raw_data.pkl")
    ).T
    train = raw_data.loc[train_index]
    val = raw_data.loc[val_index]
    noisy = raw_data.loc[noisy_index]

    return train, val, noisy


def get_only_downsampled_data():
    return get_data_split(
        run=run, data_split_artifact=data_split_artifact, data_name="only_downsampled"
    )


def get_autoencoder_denoised_data():
    autoencoder = get_model(run=run, artifact_or_name=model_artifact_name)


if __name__ == "__main__":
    sheet_name_to_evaluate_on = SheetNames.EMPATICA_LEFT_BVP.value
    data_split_version = 2
    model_artifact_name = "trained_on_EmLBVP:v0"

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type="model_evaluation",
        save_code=True,
        notes="comparison plots",
    )

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

    raw_train, raw_val, raw_noisy = get_raw_data(
        data_split_artifact=data_split_artifact,
        train_index=signal_processed_train.index,
        val_index=signal_processed_val.index,
        noisy_index=signal_processed_noisy.index,
    )
