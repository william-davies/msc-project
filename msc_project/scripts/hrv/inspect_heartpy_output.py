import os

import pandas as pd
import wandb
import heartpy
from matplotlib import pyplot as plt

from msc_project.constants import DENOISING_AUTOENCODER_PROJECT_NAME
from msc_project.scripts.hrv.get_hrv import get_artifact_dataframe


def plot_window(window_index):
    """
    Plot window across all datasets.
    :param window_index:
    :return:
    """
    plot_info = [
        (inf_raw, "inf raw"),
        (emp_raw, "emp raw"),
        (emp_traditional, "emp traditional"),
        (emp_intermediate, "emp intermediate"),
        (emp_proposed, "emp proposed denoised"),
    ]
    for heartpy_output, name in plot_info:
        heartpy.plotter(
            working_data=heartpy_output[window_index],
            measures=heartpy_output[window_index],
        )
        plt.title(name)
        plt.show()


if __name__ == "__main__":
    run_dir = "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/results/hrv_rmse/spring-sunset-337"
    filtered_dir = os.path.join(run_dir, "only_clean_inf_windows")

    inf_raw = pd.read_pickle(os.path.join(filtered_dir, "inf_raw_metrics.pkl"))
    emp_raw = pd.read_pickle(os.path.join(filtered_dir, "emp_raw_metrics.pkl"))
    emp_traditional = pd.read_pickle(
        os.path.join(filtered_dir, "emp_traditional_metrics.pkl")
    )
    emp_intermediate = pd.read_pickle(
        os.path.join(filtered_dir, "emp_intermediate_metrics.pkl")
    )
    emp_proposed = pd.read_pickle(
        os.path.join(filtered_dir, "emp_proposed_metrics.pkl")
    )

    window_index = inf_raw.columns[0]
    plot_window(window_index)

    get_hrv_version: int = 3
    upload_artifacts: bool = False

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type="hrv_evaluation",
        save_code=True,
    )

    inf_raw_data_heartpy_output = get_artifact_dataframe(
        run=run,
        artifact_or_name=f"get_hrv:v{get_hrv_version}",
        pkl_filename="inf_raw_data_hrv.pkl",
    )

    empatica_raw_data_heartpy_output = get_artifact_dataframe(
        run=run,
        artifact_or_name=f"get_hrv:v{get_hrv_version}",
        pkl_filename="empatica_raw_data_hrv.pkl",
    )

    empatica_traditional_preprocessed_data_heartpy_output = get_artifact_dataframe(
        run=run,
        artifact_or_name=f"get_hrv:v{get_hrv_version}",
        pkl_filename="empatica_traditional_preprocessed_data_hrv.pkl",
    )

    empatica_intermediate_preprocessed_data_heartpy_output = get_artifact_dataframe(
        run=run,
        artifact_or_name=f"get_hrv:v{get_hrv_version}",
        pkl_filename="empatica_intermediate_preprocessed_data_hrv.pkl",
    )

    empatica_proposed_denoised_data_heartpy_output = get_artifact_dataframe(
        run=run,
        artifact_or_name=f"get_hrv:v{get_hrv_version}",
        pkl_filename="empatica_proposed_denoised_data_hrv.pkl",
    )
    window_index = ("0720202421P1_608", "r1", "bvp", 0.0)
    plot_window(window_index)

    column = inf_raw_data_heartpy_output.iloc[:, 0]
    column = inf_raw_data_heartpy_output[window_index]
    column = empatica_raw_data_heartpy_output[window_index]
    column = empatica_traditional_preprocessed_data_heartpy_output[window_index]
    column = empatica_intermediate_preprocessed_data_heartpy_output[window_index]
    column = empatica_proposed_denoised_data_heartpy_output[window_index]

    heartpy.plotter(working_data=column, measures=column)
