"""
Make binary labels (high stress, low stress) for windows.
"""
import os

import pandas as pd
import wandb
from msc_project.constants import STRESS_PREDICTION_PROJECT_NAME
from msc_project.scripts.hrv.get_whole_signal_hrv import add_temp_file_to_artifact
from msc_project.scripts.utils import get_artifact_dataframe


def get_labels(windowed_index) -> pd.DataFrame:
    """
    Atm just works for unbalanced dataset.
    :param windowed_index:
    :return:
    """
    label_df = pd.DataFrame(
        data=False, index=windowed_index, columns=["is_high_stress"]
    )
    label_df.loc[(slice(None), "m_hard"), ["is_high_stress"]] = True
    return label_df


if __name__ == "__main__":
    hrv_features_artifact_name: str = "hrv_features:v3"
    upload_to_wandb: bool = True

    run = wandb.init(
        project=STRESS_PREDICTION_PROJECT_NAME,
        job_type="get_dataset",
        save_code=True,
    )

    raw_signal_hrv = get_artifact_dataframe(
        run=run,
        artifact_or_name=hrv_features_artifact_name,
        pkl_filename=os.path.join("changed_label", "raw_signal.pkl"),
    )

    labels = get_labels(windowed_index=raw_signal_hrv.index)

    if upload_to_wandb:
        artifact = wandb.Artifact(name="labels", type="get_dataset")
        add_temp_file_to_artifact(artifact=artifact, fp="labels.pkl", df=labels)
        run.log_artifact(artifact)
    run.finish()
