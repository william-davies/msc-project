"""
Get RMSE of HRV metrics between Infiniti GT and whatever processing I've done on Empatica.
"""
import os
from typing import Iterable, Tuple

import pandas as pd
import wandb

#%%
from msc_project.constants import DENOISING_AUTOENCODER_PROJECT_NAME, BASE_DIR
from msc_project.scripts.utils import get_artifact_dataframe

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


if __name__ == "__main__":
    hrv_artifact_name: str = "get_merged_signal_hrv:v1"
    upload_artifacts: bool = False

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type="hrv_evaluation",
        save_code=True,
    )

    inf_raw_heartpy_output = get_artifact_dataframe(
        run=run,
        artifact_or_name=hrv_artifact_name,
        pkl_filename="inf_raw.pkl",
    )

    empatica_raw_heartpy_output = get_artifact_dataframe(
        run=run,
        artifact_or_name=hrv_artifact_name,
        pkl_filename="empatica_raw.pkl",
    )

    empatica_only_downsampled_heartpy_output = get_artifact_dataframe(
        run=run,
        artifact_or_name=hrv_artifact_name,
        pkl_filename="empatica_only_downsampled.pkl",
    )

    empatica_traditional_preprocessed_heartpy_output = get_artifact_dataframe(
        run=run,
        artifact_or_name=hrv_artifact_name,
        pkl_filename="empatica_traditional_preprocessed.pkl",
    )

    empatica_dae_denoised_heartpy_output = get_artifact_dataframe(
        run=run,
        artifact_or_name=hrv_artifact_name,
        pkl_filename="empatica_dae_denoised.pkl",
    )

    inf_raw_hrv = filter_metrics_of_interest(inf_raw_heartpy_output)
    empatica_raw_hrv = filter_metrics_of_interest(empatica_raw_heartpy_output)
    empatica_only_downsampled_hrv = filter_metrics_of_interest(
        empatica_only_downsampled_heartpy_output
    )
    empatica_traditional_preprocessed_hrv = filter_metrics_of_interest(
        empatica_traditional_preprocessed_heartpy_output
    )
    empatica_dae_denoised_hrv = filter_metrics_of_interest(
        empatica_dae_denoised_heartpy_output
    )

    run_dir = os.path.join(BASE_DIR, "results", "hrv_rmse", run.name)
    to_upload_dir = os.path.join(run_dir, "to_upload")
    os.makedirs(to_upload_dir)

    hrv_data = [
        (empatica_raw_hrv, "empatica_raw"),
        (empatica_only_downsampled_hrv, "empatica_only_downsampled"),
        (
            empatica_traditional_preprocessed_hrv,
            "empatica_traditional_preprocessed",
        ),
        (empatica_dae_denoised_hrv, "empatica_dae_denoised"),
    ]

    all_rmses = get_all_rmses(gt_hrv=inf_raw_hrv, hrv_data=hrv_data)
    all_rmses.to_pickle(to_upload_dir, "all_rmses.pkl")

    for (hrv_data, data_name) in [(inf_raw_hrv, "inf_raw"), *hrv_data]:
        hrv_data.to_pickle(os.path.join(run_dir, f"{data_name}_hrv_of_interest.pkl"))

    if upload_artifacts:
        artifact = wandb.Artifact(name="hrv_rmse", type="hrv")
        artifact.add_dir(to_upload_dir)
        run.log_artifact(artifact)
    run.finish()

    empatica_raw_rmse = load_rmse(data_name="empatica_raw")
    empatica_only_downsampled_rmse = load_rmse(data_name="empatica_only_downsampled")
    empatica_traditional_preprocessed_rmse = load_rmse(
        data_name="empatica_traditional_preprocessed"
    )
    empatica_dae_denoised_rmse = load_rmse(data_name="empatica_dae_denoised")
