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
from msc_project.models.denoising_autoencoder import create_autoencoder

from wandb.keras import WandbCallback

import tensorflow as tf
import pandas as pd

# %%
from msc_project.scripts.get_data_split import DatasetPreparer


# %%
def train_autoencoder(resume: bool, train, val, epoch: int, run_id: str = ""):
    """

    :param resume: resume training a previous model
    :param train:
    :param val:
    :param epoch: how many more epochs to train.
    :param run_id: wandb run id
    :return:
    """

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-3,
        patience=1000,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    timeseries_length = train.shape[1]
    bottleneck_size = 8
    base_config = {
        "encoder_1": bottleneck_size * 2 * 2,
        "encoder_activation_1": "relu",
        "encoder_2": bottleneck_size * 2,
        "encoder_activation_2": "relu",
        "encoder_3": bottleneck_size,
        "encoder_activation_3": "relu",
        "decoder_1": bottleneck_size * 2,
        "decoder_activation_1": "relu",
        "decoder_2": bottleneck_size * 2 * 2,
        "decoder_activation_2": "relu",
        "decoder_3": timeseries_length,
        "decoder_activation_3": "sigmoid",
        "optimizer": "adam",
        "loss": "mae",
        "metric": [None],
        "batch_size": 32,
        "timeseries_length": timeseries_length,
        "monitor": "val_loss",
    }
    base_init_kwargs = {
        "project": DENOISING_AUTOENCODER_PROJECT_NAME,
        "job_type": "model_train",
    }

    if resume:
        wandb.init(
            id=run_id,
            resume="must",
            config={**base_config, "epoch": 12627 + epoch},
            force=True,
            allow_val_change=True,
            **base_init_kwargs,
        )
        best_model = wandb.restore(
            "model-best.h5", run_path=f"{DENOISING_AUTOENCODER_PROJECT_NAME}/{run_id}"
        )
        autoencoder = tf.keras.models.load_model(best_model.name)
        wandbcallback = WandbCallback(save_weights_only=False, monitor="val_loss")

        history = autoencoder.fit(
            train,
            train,
            epochs=wandb.config.epoch,
            batch_size=wandb.config.batch_size,
            validation_data=(val, val),
            callbacks=[wandbcallback, early_stop],
            shuffle=True,
            initial_epoch=wandb.run.step,
        )

    else:
        run = wandb.init(
            config={**base_config, "epoch": epoch},
            force=True,
            allow_val_change=False,
            **base_init_kwargs,
        )
        autoencoder = create_autoencoder(wandb.config)
        wandbcallback = WandbCallback(
            save_weights_only=False, monitor=wandb.config.monitor
        )

        history = autoencoder.fit(
            train,
            train,
            epochs=wandb.config.epoch,
            batch_size=wandb.config.batch_size,
            validation_data=(val, val),
            callbacks=[wandbcallback, early_stop],
            shuffle=True,
        )

    trained_model_artifact = wandb.Artifact(
        "trained_mode", type="trained_model", metadata=wandb.config
    )
    TRAINED_MODEL_DIR = os.path.join(
        BASE_DIR, "data", "preprocessed_data", "trained_model"
    )
    autoencoder.save(TRAINED_MODEL_DIR)
    trained_model_artifact.add_dir(TRAINED_MODEL_DIR)
    run.log_artifact(trained_model_artifact)
    run.finish()
    return autoencoder, history


def init_run(resume, run_id):
    # you must have both or neither
    if resume != bool(run_id):
        raise ValueError


# %%
if __name__ == "__main__":
    bottleneck_size = 8
    run_config = {
        "encoder_1": bottleneck_size * 2 * 2,
        "encoder_activation_1": "relu",
        "encoder_2": bottleneck_size * 2,
        "encoder_activation_2": "relu",
        "encoder_3": bottleneck_size,
        "encoder_activation_3": "relu",
        "decoder_1": bottleneck_size * 2,
        "decoder_activation_1": "relu",
        "decoder_2": bottleneck_size * 2 * 2,
        "decoder_activation_2": "relu",
        "decoder_activation_3": "sigmoid",
        "optimizer": "adam",
        "loss": "mae",
        "metric": [None],
        "batch_size": 32,
        "monitor": "val_loss",
        "epoch": 10,
        "patience": 1000,
        "min_delta": 1e-3,
    }

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type="model_train",
        config=run_config,
        force=True,
        allow_val_change=False,
    )

    data_split_artifact = run.use_artifact(DATA_SPLIT_ARTIFACT + ":latest")
    data_split_artifact = data_split_artifact.download(
        root=os.path.join(ARTIFACTS_ROOT, data_split_artifact.type)
    )
    train = pd.read_pickle(os.path.join(data_split_artifact, "train.pkl"))
    val = pd.read_pickle(os.path.join(data_split_artifact, "val.pkl"))

    timeseries_length = train.shape[1]
    metadata = {
        **run_config,
        "decoder_3": timeseries_length,
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
    autoencoder = create_autoencoder(metadata)

    autoencoder.fit(
        train,
        train,
        epochs=metadata["epoch"],
        batch_size=metadata["batch_size"],
        validation_data=(val, val),
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
