"""
Get RMSE of HRV metrics between Infiniti GT and whatever processing I've done on Empatica.
"""
import os
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import wandb

#%%
from msc_project.constants import DENOISING_AUTOENCODER_PROJECT_NAME, BASE_DIR
from msc_project.scripts.utils import get_committed_artifact_dataframe

# run_dir = "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/results/hrv/unique-dream-323"
# inf_raw_data_heartpy_output = pd.read_pickle(os.path.join(run_dir, "inf_raw_data_hrv.pkl"))
# empatica_raw_data_heartpy_output = pd.read_pickle(os.path.join(run_dir, "empatica_raw_data_hrv.pkl"))

# heartpy outputs a bunch of metrics. I am only interested in a subset though. mostly following Jade's previous work.
metrics_of_interest = [
    "bpm",
    "sdnn",
    "rmssd",
    "pnn50",
    "lf",
    "hf",
    "lf/hf",
]


def filter_metrics_of_interest(heartpy_output: pd.DataFrame) -> pd.DataFrame:
    return heartpy_output.loc[metrics_of_interest]


def load_rmse(data_name) -> pd.DataFrame:
    return pd.read_pickle(os.path.join(to_upload_dir, f"{data_name}_rmse.pkl"))


# %%
def get_dtypes_of_series(series: pd.Series):
    for index, row in series.iteritems():
        print(f"{index}: {type(row)}")


def get_all_rmses(gt_hrv: pd.DataFrame, hrv_data: Iterable[Tuple]) -> pd.DataFrame:
    """

    :param gt_hrv: from Infiniti raw
    :param hrv_data: (DataFrame, name), ...
    :return:
    """
    rmses = []
    for hrv, _ in hrv_data:
        rmse = get_rmse(gt_hrv_metrics=gt_hrv, other_hrv_metrics=hrv)
        rmses.append(rmse)

    all_rmses = pd.concat(
        objs=rmses,
        axis=1,
        keys=[hrv_data[1] for hrv_data in hrv_data],
    )
    return all_rmses


def get_rmse(
    gt_hrv_metrics: pd.DataFrame, other_hrv_metrics: pd.DataFrame
) -> pd.DataFrame:
    """
    Get root mean square error. `gt_hrv_metrics` and `other_hrv_metrics` must have exactly the same shape, index, columns.
    :param gt_hrv_metrics: rows is hrv metric. columns is multiindex window.
    :param other_hrv_metrics:
    :return:
    """
    delta = gt_hrv_metrics - other_hrv_metrics
    mean_square = (delta ** 2).mean(axis=1)
    rmse = mean_square ** 0.5
    return rmse


def get_hrv(pkl_filename: str, clean_indexes: pd.Index) -> pd.DataFrame:
    """
    Filter hrv metrics of interest. Also only select the indexes (clean indexes) that we're using for the RMSE calculation.
    :param pkl_filename: as uploaded to Wandb
    :param clean_indexes: Infiniti GT is clean at these indexes
    :return:
    """
    heartpy_output = get_committed_artifact_dataframe(
        run=run,
        artifact_or_name=hrv_artifact_name,
        pkl_filename=pkl_filename,
    )
    hrv = filter_metrics_of_interest(heartpy_output)
    return hrv[clean_indexes]


def get_clean_signal_indexes(
    noisy_mask: pd.DataFrame, noise_tolerance: float = 0
) -> pd.Index:
    """
    Return indexes of signals that are below noise tolerance.
    :param noisy_mask: binary. rows is TimedeltaIndex.
    :param noise_tolerance: between 0 and 1
    :return:
    """
    noisy_proportions = get_noisy_proportions(noisy_mask)
    is_clean = noisy_proportions <= noise_tolerance
    clean_indexes = is_clean.index[is_clean]
    return clean_indexes


def get_noisy_proportions(noisy_mask):
    """
    Rows is TimedeltaIndex.
    :param noisy_mask:
    :return:
    """
    return noisy_mask.sum(axis=0) / noisy_mask.shape[0]


def heartpy_failed(hrv):
    """
    Returns boolean series.
    :param hrv:
    :return:
    """
    return hrv.isna().any(axis=0)


def heartpy_succeeded(hrv):
    """
    Returns boolean series.
    :param hrv:
    :return:
    """
    return hrv.notna().any(axis=0)


def assert_all_indexes_equal(indexes):
    reference = indexes[0]
    for index in indexes[1:]:
        assert reference.equals(index)


if __name__ == "__main__":
    hrv_artifact_name: str = "get_merged_signal_hrv:v2"
    preprocessed_data_artifact_name: str = "Inf_preprocessed_data:v8"
    upload_artifacts: bool = False
    noise_tolerance: float = 0
    config = {"noise_tolerance": noise_tolerance}

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type="hrv_evaluation",
        save_code=True,
        config=config,
    )

    noisy_mask = get_committed_artifact_dataframe(
        run=run,
        artifact_or_name=preprocessed_data_artifact_name,
        pkl_filename=os.path.join("not_windowed", "noisy_mask.pkl"),
    )
    clean_indexes = get_clean_signal_indexes(
        noisy_mask=noisy_mask, noise_tolerance=noise_tolerance
    )

    inf_raw_hrv = get_hrv(pkl_filename="inf_raw.pkl", clean_indexes=clean_indexes)
    empatica_raw_hrv = get_hrv(
        pkl_filename="empatica_raw.pkl", clean_indexes=clean_indexes
    )
    empatica_only_downsampled_hrv = get_hrv(
        pkl_filename="empatica_only_downsampled.pkl", clean_indexes=clean_indexes
    )
    empatica_traditional_preprocessed_hrv = get_hrv(
        pkl_filename="empatica_traditional_preprocessed.pkl",
        clean_indexes=clean_indexes,
    )
    empatica_dae_denoised_hrv = get_hrv(
        pkl_filename="empatica_dae_denoised.pkl", clean_indexes=clean_indexes
    )

    all_hrvs = [
        inf_raw_hrv,
        empatica_raw_hrv,
        empatica_only_downsampled_hrv,
        empatica_traditional_preprocessed_hrv,
        empatica_dae_denoised_hrv,
    ]
    assert_all_indexes_equal(indexes=[hrv.columns for hrv in all_hrvs])
    assert_all_indexes_equal(indexes=[hrv.index for hrv in all_hrvs])

    # remove windows where heartpy failed
    heartpy_succeededs = [heartpy_succeeded(hrv) for hrv in all_hrvs]
    heartpy_available_for_all = inf_raw_hrv.columns[
        np.logical_and.reduce(heartpy_succeededs)
    ]

    inf_raw_hrv_no_nans = inf_raw_hrv[heartpy_available_for_all]
    empatica_raw_hrv_no_nans = empatica_raw_hrv[heartpy_available_for_all]
    empatica_only_downsampled_hrv_no_nans = empatica_only_downsampled_hrv[
        heartpy_available_for_all
    ]
    empatica_traditional_preprocessed_hrv_no_nans = (
        empatica_traditional_preprocessed_hrv[heartpy_available_for_all]
    )
    empatica_dae_denoised_hrv_no_nans = empatica_dae_denoised_hrv[
        heartpy_available_for_all
    ]

    run_dir = os.path.join(BASE_DIR, "results", "hrv_rmse", run.name)
    to_upload_dir = os.path.join(run_dir, "to_upload")
    os.makedirs(to_upload_dir)

    hrv_data = [
        (empatica_raw_hrv_no_nans, "empatica_raw"),
        (empatica_only_downsampled_hrv_no_nans, "empatica_only_downsampled"),
        (
            empatica_traditional_preprocessed_hrv_no_nans,
            "empatica_traditional_preprocessed",
        ),
        (empatica_dae_denoised_hrv_no_nans, "empatica_dae_denoised"),
    ]

    all_rmses = get_all_rmses(gt_hrv=inf_raw_hrv_no_nans, hrv_data=hrv_data)
    all_rmses.to_pickle(os.path.join(to_upload_dir, "all_rmses.pkl"))
    all_rmses.to_csv(os.path.join(to_upload_dir, "all_rmses.csv"))

    for (hrv_data, data_name) in [(inf_raw_hrv_no_nans, "inf_raw"), *hrv_data]:
        hrv_data.to_pickle(os.path.join(run_dir, f"{data_name}_hrv_of_interest.pkl"))

    if upload_artifacts:
        artifact = wandb.Artifact(name="hrv_rmse", type="hrv", metadata=config)
        artifact.add_dir(to_upload_dir)
        run.log_artifact(artifact)
    run.finish()
