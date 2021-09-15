import os

import wandb
from wandb.integration.keras import WandbCallback

from msc_project.constants import (
    SheetNames,
    TRAINED_MODEL_ARTIFACT,
    BASE_DIR,
    STRESS_PREDICTION_PROJECT_NAME,
)
from msc_project.models.stress_prediction.mlp import instantiate_predictor
from msc_project.scripts.hrv.get_hrv import get_artifact_dataframe
from msc_project.scripts.train_autoencoder import (
    init_run,
    get_model,
    get_initial_epoch,
    save_model,
)
import tensorflow as tf

if __name__ == "__main__":
    sheet_name = SheetNames.INFINITY.value
    data_split_version = 0
    notes = ""
    run_id = ""
    data_name: str = "only_downsampled"

    run_config = {
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "metric": [None],
        "batch_size": 32,
        "monitor": "val_loss",
        "epoch": 5,
        "patience": 1000,
        "min_delta": 1e-3,
        "data_name": data_name,
    }

    run = init_run(
        run_config=run_config,
        run_id=run_id,
        notes=notes,
        project=STRESS_PREDICTION_PROJECT_NAME,
    )

    data_split_artifact = run.use_artifact(
        artifact_or_name=f"{sheet_name}_data_split:v{data_split_version}"
    )
    train_X = get_artifact_dataframe(
        run=run,
        artifact_or_name=data_split_artifact,
        pkl_filename=os.path.join(data_name, "train.pkl"),
    )
    val_X = get_artifact_dataframe(
        run=run,
        artifact_or_name=data_split_artifact,
        pkl_filename=os.path.join(data_name, "val.pkl"),
    )
    train_y = get_artifact_dataframe(
        run=run,
        artifact_or_name=data_split_artifact,
        pkl_filename=os.path.join("labels", "train.pkl"),
    )
    val_y = get_artifact_dataframe(
        run=run,
        artifact_or_name=data_split_artifact,
        pkl_filename=os.path.join("labels", "val.pkl"),
    )

    timeseries_length = train_X.shape[1]
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
    predictor = get_model(
        run=run, config=metadata, model_instantiator=instantiate_predictor
    )
    print(f"predictor.summary(): {predictor.summary()}")

    history = predictor.fit(
        train_X,
        train_y,
        epochs=metadata["epoch"],
        initial_epoch=get_initial_epoch(run),
        batch_size=metadata["batch_size"],
        validation_data=(
            val_X,
            val_y,
        ),
        callbacks=[wandbcallback, early_stop],
        shuffle=True,
    )

    trained_model_artifact = wandb.Artifact(
        name=f"trained_on_{data_name}", type=TRAINED_MODEL_ARTIFACT, metadata=metadata
    )
    trained_model_dir = os.path.join(
        BASE_DIR, "data", "stress_prediction", TRAINED_MODEL_ARTIFACT
    )
    os.makedirs(trained_model_dir)
    save_model(save_path=trained_model_dir, model=predictor)
    trained_model_artifact.add_dir(trained_model_dir)
    run.log_artifact(trained_model_artifact)
    run.finish()
