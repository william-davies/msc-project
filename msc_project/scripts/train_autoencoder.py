import json
import os

import wandb

from msc_project.constants import (
    DENOISING_AUTOENCODER_PROJECT_NAME,
    DATA_SPLIT_ARTIFACT,
    ARTIFACTS_ROOT,
    BASE_DIR,
    TRAINED_MODEL_ARTIFACT,
)
from msc_project.models.lstm_autoencoder import create_autoencoder, reshape_data

from wandb.keras import WandbCallback

import tensorflow as tf
import pandas as pd

# %%
def init_run(resume: bool, run_config, run_id: str = ""):
    # you must have both or neither
    if resume != bool(run_id):
        raise ValueError

    if resume:
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


def load_data(run, data_split_version: int):
    """

    :param run: so we can "use_artifact" in wandb artifact graph
    :param data_split_version:
    :return:
    """
    data_split_artifact = run.use_artifact(
        DATA_SPLIT_ARTIFACT + f":v{data_split_version}"
    )
    data_split_artifact = data_split_artifact.download(
        root=os.path.join(ARTIFACTS_ROOT, data_split_artifact.type)
    )
    train = pd.read_pickle(os.path.join(data_split_artifact, "train.pkl"))
    val = pd.read_pickle(os.path.join(data_split_artifact, "val.pkl"))
    return train, val


def get_autoencoder(run, metadata):
    if run.resumed:
        best_model = wandb.restore("model-best.h5", run_path=run.path)
        autoencoder = tf.keras.models.load_model(best_model.name)
    else:
        autoencoder = create_autoencoder(metadata)
    return autoencoder


# %%
if __name__ == "__main__":
    run_config = {
        "optimizer": "adam",
        "loss": "mae",
        "metric": [None],
        "batch_size": 32,
        "monitor": "val_loss",
        "epoch": 5000,
        "patience": 1500,
        "min_delta": 1e-3,
    }

    resume = True
    run_id = "2zf4vtrx"

    run = init_run(resume=resume, run_config=run_config, run_id=run_id)

    train, val = load_data(run=run, data_split_version=7)

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

    history = autoencoder.fit(
        reshape_data(train),
        reshape_data(train),
        epochs=metadata["epoch"],
        batch_size=metadata["batch_size"],
        validation_data=(reshape_data(val), reshape_data(val)),
        callbacks=[wandbcallback, early_stop],
        shuffle=True,
    )

    trained_model_artifact = wandb.Artifact(
        TRAINED_MODEL_ARTIFACT, type=TRAINED_MODEL_ARTIFACT, metadata=metadata
    )
    TRAINED_MODEL_DIR = os.path.join(
        BASE_DIR, "data", "preprocessed_data", TRAINED_MODEL_ARTIFACT
    )
    autoencoder.save(TRAINED_MODEL_DIR)
    trained_model_artifact.add_dir(TRAINED_MODEL_DIR)
    run.log_artifact(trained_model_artifact)
    run.finish()
