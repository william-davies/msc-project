import os
from typing import Iterable

import pandas as pd
import wandb
import numpy as np

#%%
from msc_project.constants import DENOISING_AUTOENCODER_PROJECT_NAME

# run_dir = "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/results/hrv/unique-dream-323"
# inf_raw_data_heartpy_output = pd.read_pickle(os.path.join(run_dir, "inf_raw_data_hrv.pkl"))
# empatica_raw_data_heartpy_output = pd.read_pickle(os.path.join(run_dir, "empatica_raw_data_hrv.pkl"))
from msc_project.scripts.hrv.get_rmse import get_all_rmses
from msc_project.scripts.utils import get_artifact_dataframe

metrics_of_interest = [
    "bpm",
    "ibi",
    "sdnn",
    "rmssd",
    "pnn20",
    "pnn50",
    "hf",
    "sdsd",
    "hr_mad",
]


def get_hrv_metrics(heartpy_output: pd.DataFrame) -> pd.DataFrame:
    return heartpy_output.loc[metrics_of_interest]


# %%


def read_hrv_of_interest(filename) -> pd.DataFrame:
    return replace_masked_with_nan(pd.read_pickle(os.path.join(run_dir, filename)))


def replace_masked_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    cond = df.applymap(lambda element: element is np.ma.masked)
    replaced = df.mask(cond=cond)
    return replaced


def get_nans_per_metric(metric_set: pd.DataFrame) -> pd.Series:
    """
    Sometimes heartpy fails to compute a metric. Often when there are no consecutive differences (i.e. no consecutive 3 detected peaks).
    :return:
    """
    isna = metric_set.isna()
    return isna.sum(axis=1)


def get_clean_signal_indexes(noisy_mask: pd.DataFrame, noise_tolerance: float = 0):
    """
    Return indexes of signals that are below noise tolerance.
    :param noisy_mask: binary
    :param noise_tolerance: between 0 and 1
    :return:
    """
    noisy_proportions = noisy_mask.sum(axis=0) / noisy_mask.shape[0]
    is_clean = noisy_proportions <= noise_tolerance
    clean_indexes = is_clean.index[is_clean]
    return clean_indexes


def get_all_nan_counts(metric_sets: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """

    :param metric_sets:
    :return:
    """
    nan_counts = []
    for metric_set in metric_sets:
        nan_count = get_nans_per_metric(metric_set=metric_set)
        nan_counts.append(nan_count)

    all_nan_counts = pd.concat(
        objs=nan_counts,
        axis=1,
        keys=[
            "inf_raw",
            "empatica_raw",
            "empatica_traditional_preprocessed",
            "empatica_intermediate_preprocessed",
            "empatica_proposed_denoised",
        ],
    )
    return all_nan_counts


def get_normalized_delta(df1, df2):
    delta = df1 - df2
    normalized = delta.divide((df1 + df2) / 2)
    return normalized


if __name__ == "__main__":
    rmse_artifact_name: str = "hrv_rmse:v2"

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type="hrv_evaluation",
        save_code=True,
    )

    empatica_raw_rmse = get_artifact_dataframe(
        run=run,
        artifact_or_name=rmse_artifact_name,
        pkl_filename="empatica_raw_rmse.pkl",
    )

    # %%
    inf_windowed_noisy_mask = pd.read_pickle(
        "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/msc_project/scripts/wandb_artifacts/preprocessed_data/inf_preprocessed_datav2/windowed_noisy_mask.pkl"
    )
    clean_window_indexes = get_clean_signal_indexes(noisy_mask=inf_windowed_noisy_mask)

    emp_windowed_noisy_mask = pd.read_pickle(
        "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/msc_project/scripts/wandb_artifacts/preprocessed_data/emlbvp_preprocessed_datav3/windowed_noisy_mask.pkl"
    )
    emp_clean_window_indexes = get_clean_signal_indexes(
        noisy_mask=emp_windowed_noisy_mask
    )

    # %%
    filtered_inf_raw = inf_raw[clean_window_indexes]
    filtered_emp_raw = emp_raw[clean_window_indexes]
    filtered_emp_traditional = emp_traditional[clean_window_indexes]
    filtered_emp_intermediate = emp_intermediate[clean_window_indexes]
    filtered_emp_proposed = emp_proposed[clean_window_indexes]

    # %%
    filtered_dir = os.path.join(run_dir, "only_clean_inf_windows")

    filtered_inf_raw.to_pickle(os.path.join(filtered_dir, "inf_raw_metrics.pkl"))
    filtered_emp_raw.to_pickle(os.path.join(filtered_dir, "emp_raw_metrics.pkl"))
    filtered_emp_traditional.to_pickle(
        os.path.join(filtered_dir, "emp_traditional_metrics.pkl")
    )
    filtered_emp_intermediate.to_pickle(
        os.path.join(filtered_dir, "emp_intermediate_metrics.pkl")
    )
    filtered_emp_proposed.to_pickle(
        os.path.join(filtered_dir, "emp_proposed_metrics.pkl")
    )

    # %%
    # os.makedirs(filtered_dir)
    filtered_all_nan_counts = get_all_nan_counts(
        metric_sets=(
            filtered_inf_raw,
            filtered_emp_raw,
            filtered_emp_traditional,
            filtered_emp_intermediate,
            filtered_emp_proposed,
        )
    )
    filtered_all_nan_counts.to_pickle(os.path.join(filtered_dir, "all_nan_counts.pkl"))

    filtered_all_rmses = get_all_rmses(
        hrv_data=(
            filtered_emp_raw,
            filtered_emp_traditional,
            filtered_emp_intermediate,
            filtered_emp_proposed,
        )
    )
    # filtered_all_rmses.to_pickle(os.path.join(filtered_dir, "all_rmses.pkl"))

    # %%

    not_filtered_all_nan_counts = get_all_nan_counts(
        metric_sets=(inf_raw, emp_raw, emp_traditional, emp_intermediate, emp_proposed)
    )
    not_filtered_all_nan_counts.to_pickle(os.path.join(run_dir, "all_nan_counts.pkl"))

    not_filtered_all_rmses = get_all_rmses(
        hrv_data=(emp_raw, emp_traditional, emp_intermediate, emp_proposed)
    )
    not_filtered_all_rmses.to_pickle(os.path.join(run_dir, "all_rmses.pkl"))
