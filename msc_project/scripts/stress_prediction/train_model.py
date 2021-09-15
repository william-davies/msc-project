import os

from msc_project.constants import SheetNames
from msc_project.scripts.hrv.get_hrv import get_artifact_dataframe
from msc_project.scripts.train_autoencoder import init_run


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
        "epoch": 100,
        "patience": 1000,
        "min_delta": 1e-3,
    }

    run = init_run(run_config=run_config, run_id=run_id, notes=notes)

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
