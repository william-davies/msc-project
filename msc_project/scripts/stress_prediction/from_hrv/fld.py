"""
Get FLD
"""
import os

import numpy as np
import wandb

from msc_project.constants import STRESS_PREDICTION_PROJECT_NAME, SheetNames
from msc_project.scripts.utils import get_committed_artifact_dataframe


def get_fld_wrapper(sheet_name, feature_set_name):
    def get_fld(low_stress, high_stress):
        numerator = low_stress.mean() - high_stress.mean()
        numerator = np.abs(numerator)
        denominator = low_stress.std() ** 2 + high_stress.std() ** 2
        return numerator / denominator

    pkl_filename = os.path.join(sheet_name, f"{feature_set_name}_hrv_features.pkl")
    hrv_features = get_committed_artifact_dataframe(
        artifact_or_name=complete_dataset_artifact_name,
        pkl_filename=pkl_filename,
        run=run,
    )
    low_stress = hrv_features.loc[stress_labels.index[~stress_labels["is_high_stress"]]]
    high_stress = hrv_features.loc[stress_labels.index[stress_labels["is_high_stress"]]]
    fld = get_fld(low_stress=low_stress, high_stress=high_stress)
    return fld


if __name__ == "__main__":
    complete_dataset_artifact_name: str = "complete_dataset:v25"

    run = wandb.init(
        project=STRESS_PREDICTION_PROJECT_NAME,
        job_type="get_dataset",
        save_code=True,
    )

    stress_labels = get_committed_artifact_dataframe(
        run=run,
        artifact_or_name=complete_dataset_artifact_name,
        pkl_filename="stress_labels.pkl",
    )

    fld = get_fld_wrapper(
        sheet_name=SheetNames.EMPATICA_LEFT_BVP.value, feature_set_name="raw_signal"
    )
