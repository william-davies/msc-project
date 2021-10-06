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

from msc_project.models.lstm_autoencoder import create_autoencoder

from wandb.keras import WandbCallback

import tensorflow as tf
import pandas as pd

# %%
from msc_project.scripts.evaluate_autoencoder import data_has_num_features_dimension
from msc_project.scripts.utils import add_num_features_dimension, get_artifact_dataframe


def init_run(
    run_config,
    run_id: str = "",
    notes: str = "",
    project=DENOISING_AUTOENCODER_PROJECT_NAME,
):
    """

    :param run_config:
    :param run_id: of run you want to resume
    :return:
    """
    base_init_kwargs = {"save_code": True, "notes": notes}
    if run_id:
        run = wandb.init(
            id=run_id,
            resume="must",
            project=project,
            job_type="model_train",
            config=run_config,
            force=True,
            allow_val_change=False,
            **base_init_kwargs,
        )
    else:
        run = wandb.init(
            resume="never",
            project=project,
            job_type="model_train",
            config=run_config,
            force=True,
            allow_val_change=False,
            **base_init_kwargs,
        )
    return run


def get_model(run, config, model_instantiator=create_autoencoder):
    """
    Returns either partially trained model for resumed run, or brand new model.
    :param run:
    :param config:
    :param model_instantiator: instantiates new model
    :return:
    """
    if run.resumed:
        best_model = wandb.restore("model-best.h5", run_path=run.path)
        model = tf.keras.models.load_model(best_model.name)
    else:
        model = model_instantiator(config)
    return model


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


def save_model(save_path: str, model):
    """
    Deletes save destination directory first.
    :param save_path:
    :param model:
    :return:
    """
    shutil.rmtree(path=save_path, ignore_errors=True)
    model.save(save_path)


# %%
if __name__ == "__main__":
    sheet_name = SheetNames.INFINITY.value
    data_split_version = 5
    data_name: str = "only_downsampled"
    run_id = ""
    architecture_type = get_architecture_type(create_autoencoder)
    notes = f"{architecture_type} trained on {sheet_name} {data_name}"
    upload_artifact: bool = False

    run_config = {
        "optimizer": "adam",
        "loss": "mae",
        "metric": [None],
        "batch_size": 32,
        "monitor": "val_loss",
        "epoch": 10,
        "patience": 1000,
        "min_delta": 1e-3,
        "model_architecture_type": architecture_type,
        "sheet_name": sheet_name,
        "data_name": data_name,
    }

    run = init_run(
        run_config=run_config,
        run_id=run_id,
        notes=notes,
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
    )

    data_split_artifact = run.use_artifact(
        artifact_or_name=f"{sheet_name}_data_split:v{data_split_version}"
    )
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
    autoencoder = get_model(
        run=run, config=metadata, model_instantiator=create_autoencoder
    )
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

    if upload_artifact:
        trained_model_artifact = wandb.Artifact(
            name=f"trained_on_{sheet_name}",
            type=TRAINED_MODEL_ARTIFACT,
            metadata=metadata,
            description=notes,
        )
        trained_model_dir = os.path.join(
            BASE_DIR, "data", DENOISING_AUTOENCODER_PROJECT_NAME, TRAINED_MODEL_ARTIFACT
        )
        save_model(save_path=trained_model_dir, model=autoencoder)
        trained_model_artifact.add_dir(trained_model_dir)
        run.log_artifact(trained_model_artifact)
    run.finish()
