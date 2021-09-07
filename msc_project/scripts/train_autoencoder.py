import json
import os
import shutil

import wandb

from msc_project.constants import (
    DENOISING_AUTOENCODER_PROJECT_NAME,
    DATA_SPLIT_ARTIFACT,
    ARTIFACTS_ROOT,
    BASE_DIR,
    TRAINED_MODEL_ARTIFACT,
    SheetNames,
)

from msc_project.models.mlp_autoencoder import create_autoencoder

from wandb.keras import WandbCallback

import tensorflow as tf
import pandas as pd

# %%
from msc_project.scripts.evaluate_autoencoder import data_has_num_features_dimension
from msc_project.scripts.utils import add_num_features_dimension


def init_run(run_config, run_id: str = ""):
    """

    :param run_config:
    :param run_id: of run you want to resume
    :return:
    """
    if run_id:
        run = wandb.init(
            id=run_id,
            resume="must",
            project=DENOISING_AUTOENCODER_PROJECT_NAME,
            job_type="model_train",
            config=run_config,
            force=True,
            allow_val_change=False,
        )
    else:
        run = wandb.init(
            resume="never",
            project=DENOISING_AUTOENCODER_PROJECT_NAME,
            job_type="model_train",
            config=run_config,
            force=True,
            allow_val_change=False,
        )
    return run


def load_data(data_split_artifact):
    """

    :param data_split_artifact:
    :return:
    """
    data_split_artifact = data_split_artifact.download(
        root=os.path.join(ARTIFACTS_ROOT, data_split_artifact.type)
    )
    train = pd.read_pickle(os.path.join(data_split_artifact, "train.pkl"))
    val = pd.read_pickle(os.path.join(data_split_artifact, "val.pkl"))
    return train, val


def get_autoencoder(run, metadata):
    """
    Returns either partially trained autoencoder for resumed run, or brand new autoencoder.
    :param run:
    :param metadata:
    :return:
    """
    if run.resumed:
        best_model = wandb.restore("model-best.h5", run_path=run.path)
        autoencoder = tf.keras.models.load_model(best_model.name)
    else:
        autoencoder = create_autoencoder(metadata)
    return autoencoder


def get_initial_epoch(run):
    """
    If we resume a run, we load the model from the best_epoch. This best_epoch is the initial_epoch.
    This leads to some confusion because the wandb step just keeps incrementing by 1 every epoch. But it does the job for now.
    :param run:
    :return:
    """
    if run.resumed:
        return run.summary["best_epoch"]
    else:
        return 0


def get_architecture_type(create_autoencoder):
    module = create_autoencoder.__module__
    return module.split(".")[-1]


# %%
if __name__ == "__main__":
    sheet_name = SheetNames.INFINITY.value
    data_split_version = 0

    is_production: bool = False
    run_config = {
        "optimizer": "adam",
        "loss": "mae",
        "metric": [None],
        "batch_size": 32,
        "monitor": "val_loss",
        "epoch": 30,
        "patience": 1500,
        "min_delta": 1e-3,
        "model_architecture_type": get_architecture_type(create_autoencoder),
        "is_prod": is_production,
    }

    run_id = ""
    run = init_run(run_config=run_config, run_id=run_id)

    data_split_artifact = run.use_artifact(
        artifact_or_name=f"{sheet_name}_data_split:v{data_split_version}"
    )
    train, val = load_data(data_split_artifact=data_split_artifact)

    timeseries_length = train.shape[1]
    metadata = {
        **run_config,
        "timeseries_length": timeseries_length,
    }
    wandbcallback = WandbCallback(save_weights_only=False, monitor=metadata["monitor"])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=metadata["monitor"],
        min_delta=metadata["min_delta"],
        patience=metadata["patience"],
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )
    autoencoder = get_autoencoder(run=run, metadata=metadata)
    print(autoencoder.summary())

    if data_has_num_features_dimension(autoencoder):
        train = add_num_features_dimension(train)
        val = add_num_features_dimension(val)

    history = autoencoder.fit(
        train,
        train,
        epochs=metadata["epoch"],
        initial_epoch=get_initial_epoch(run),
        batch_size=metadata["batch_size"],
        validation_data=(
            val,
            val,
        ),
        callbacks=[wandbcallback, early_stop],
        shuffle=True,
    )

    trained_model_artifact = wandb.Artifact(
        TRAINED_MODEL_ARTIFACT, type=TRAINED_MODEL_ARTIFACT, metadata=metadata
    )
    TRAINED_MODEL_DIR = os.path.join(
        BASE_DIR, "data", "preprocessed_data", TRAINED_MODEL_ARTIFACT
    )
    shutil.rmtree(path=TRAINED_MODEL_DIR, ignore_errors=True)
    autoencoder.save(TRAINED_MODEL_DIR)
    trained_model_artifact.add_dir(TRAINED_MODEL_DIR)
    run.log_artifact(trained_model_artifact)
    run.finish()
