import os

import pandas as pd
import wandb

from msc_project.constants import SheetNames, DENOISING_AUTOENCODER_PROJECT_NAME
from msc_project.scripts.evaluate_autoencoder import (
    get_model,
    download_preprocessed_data,
    get_reconstructed_df,
    plot_examples,
)
from msc_project.scripts.utils import get_artifact_dataframe


def get_signal_processed_data(run, data_split_artifact):
    train, val, noisy = get_data_split(
        run=run,
        data_split_artifact=data_split_artifact,
        data_name="intermediate_preprocessed",
    )
    signal_processed = pd.concat(objs=(train, val, noisy))
    return signal_processed


def get_data_split_indexes(run, data_split_artifact):
    train, val, noisy = get_only_downsampled_data(
        run=run, data_split_artifact=data_split_artifact
    )
    indexes = tuple(map(lambda split: split.index, (train, val, noisy)))
    return indexes


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


def get_raw_data(data_split_artifact):
    preprocessed_data_fp = download_preprocessed_data(data_split_artifact)
    # transpose so examples is row axis. like train/val/noisy
    raw_data = pd.read_pickle(
        os.path.join(preprocessed_data_fp, "windowed_raw_data.pkl")
    ).T

    return raw_data


def get_only_downsampled_data(run, data_split_artifact):
    return get_data_split(
        run=run, data_split_artifact=data_split_artifact, data_name="only_downsampled"
    )


def get_autoencoder_reconstructed_data(run, model_artifact_name, data_split_artifact):
    downsampled_train, downsampled_val, downsampled_noisy = get_only_downsampled_data(
        run=run, data_split_artifact=data_split_artifact
    )
    downsampled = pd.concat(
        objs=(downsampled_train, downsampled_val, downsampled_noisy)
    )
    autoencoder = get_model(run=run, artifact_or_name=model_artifact_name)

    reconstructed = get_reconstructed_df(
        to_reconstruct=downsampled, autoencoder=autoencoder
    )

    return reconstructed


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

    signal_processed = get_signal_processed_data(
        run=run, data_split_artifact=data_split_artifact
    )

    raw_data = get_raw_data(
        data_split_artifact=data_split_artifact,
    )

    reconstructed = get_autoencoder_reconstructed_data(
        run=run,
        model_artifact_name=model_artifact_name,
        data_split_artifact=data_split_artifact,
    )

    train_index, val_index, noisy_index = get_data_split_indexes(
        run=run, data_split_artifact=data_split_artifact
    )

    datasets_to_plot = [
        (raw_data, {"color": "k", "label": "original signal"}),
        (
            intermediate_preprocessed,
            {"color": "b", "label": "ae input"},
        ),
        (reconstructed, {"color": "g", "label": "reconstructed"}),
    ]

    plot_examples(
        run_name=run.name, example_type="train", datasets_to_plot=datasets_to_plot
    )
