import os

import numpy as np
import pandas as pd
import wandb
import heartpy
from matplotlib import pyplot as plt

from msc_project.constants import DENOISING_AUTOENCODER_PROJECT_NAME
from msc_project.scripts.evaluate_autoencoder import show_or_save
from msc_project.scripts.hrv.get_hrv import get_artifact_dataframe
from msc_project.scripts.hrv.hrv_testing_ground import get_clean_window_indexes
from msc_project.scripts.utils import slugify


def plot_window(window_index):
    """
    Plot window across all datasets.
    :param window_index:
    :return:
    """
    plot_info = [
        (inf_raw_data_heartpy_output, "inf raw"),
        (empatica_raw_data_heartpy_output, "emp raw"),
        (empatica_traditional_preprocessed_data_heartpy_output, "emp traditional"),
        (empatica_intermediate_preprocessed_data_heartpy_output, "emp intermediate"),
        (empatica_proposed_denoised_data_heartpy_output, "emp proposed denoised"),
    ]
    window_dir = os.path.join(signal_plots_dir, slugify(window_index))
    os.makedirs(window_dir)
    for heartpy_output, name in plot_info:
        # heartpy.plotter(
        #     working_data=heartpy_output[window_index],
        #     measures=heartpy_output[window_index],
        #     title=f'{name}\n{window_index}',
        # )
        heartpy_plotter_wrapper(
            working_data=heartpy_output[window_index],
            measures=heartpy_output[window_index],
            title=f"{name}\n{window_index}",
            show=False,
            save_filepath=os.path.join(window_dir, name),
        )


@show_or_save
def heartpy_plotter_wrapper(*args, **kwargs):
    heartpy.plotter(*args, **kwargs)


if __name__ == "__main__":
    run_dir = "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/results/hrv_rmse/spring-sunset-337"
    filtered_dir = os.path.join(run_dir, "only_clean_inf_windows")

    get_hrv_version: int = 3
    upload_artifacts: bool = False

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type="hrv_evaluation",
        save_code=True,
    )

    inf_windowed_noisy_mask = pd.read_pickle(
        "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/msc_project/scripts/wandb_artifacts/preprocessed_data/inf_preprocessed_datav2/windowed_noisy_mask.pkl"
    )
    clean_window_indexes = get_clean_window_indexes(
        windowed_noisy_mask=inf_windowed_noisy_mask
    )

    inf_raw_data_heartpy_output = get_artifact_dataframe(
        run=run,
        artifact_or_name=f"get_hrv:v{get_hrv_version}",
        pkl_filename="inf_raw_data_hrv.pkl",
    )[clean_window_indexes]

    empatica_raw_data_heartpy_output = get_artifact_dataframe(
        run=run,
        artifact_or_name=f"get_hrv:v{get_hrv_version}",
        pkl_filename="empatica_raw_data_hrv.pkl",
    )[clean_window_indexes]

    empatica_traditional_preprocessed_data_heartpy_output = get_artifact_dataframe(
        run=run,
        artifact_or_name=f"get_hrv:v{get_hrv_version}",
        pkl_filename="empatica_traditional_preprocessed_data_hrv.pkl",
    )[clean_window_indexes]

    empatica_intermediate_preprocessed_data_heartpy_output = get_artifact_dataframe(
        run=run,
        artifact_or_name=f"get_hrv:v{get_hrv_version}",
        pkl_filename="empatica_intermediate_preprocessed_data_hrv.pkl",
    )[clean_window_indexes]

    empatica_proposed_denoised_data_heartpy_output = get_artifact_dataframe(
        run=run,
        artifact_or_name=f"get_hrv:v{get_hrv_version}",
        pkl_filename="empatica_proposed_denoised_data_hrv.pkl",
    )[clean_window_indexes]

    signal_plots_dir = os.path.join(filtered_dir, "signal_plots")
    # os.makedirs(signal_plots_dir)

    num_examples_to_plot: int = 20
    window_indexes = np.random.choice(
        a=inf_raw_data_heartpy_output.columns, size=num_examples_to_plot, replace=False
    )
    for window_index in window_indexes:
        plot_window(window_index)
