"""
Check if downsample first works properly.
"""
import os

import wandb
from matplotlib import pyplot as plt

from msc_project.constants import DENOISING_AUTOENCODER_PROJECT_NAME, SECONDS_IN_MINUTE
from msc_project.scripts.get_preprocessed_data import (
    normalize_windows,
    get_temporal_subwindow_of_signal,
)
from msc_project.scripts.plot_raw_data import plot_signal
from msc_project.scripts.utils import get_artifact_dataframe


def plot_examples(datasets_to_plot, windows_to_plot):
    """
    For non windowed data.
    :param datasets_to_plot:
    :param windows_to_plot:
    :return:
    """
    for window_index in windows_to_plot:
        for (data, plot_kwargs) in datasets_to_plot:
            signal = data[window_index]
            plot_signal(signal=signal, **plot_kwargs)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    preprocessed_data_artifact_name = "EmLBVP_preprocessed_data:v6"
    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type="compare_processing_methods",
        save_code=True,
    )
    raw_data = get_artifact_dataframe(
        run=run,
        artifact_or_name=preprocessed_data_artifact_name,
        pkl_filename=os.path.join("not_windowed", "raw_data.pkl"),
    )
    raw_data = get_temporal_subwindow_of_signal(
        df=raw_data, window_start=SECONDS_IN_MINUTE, window_end=4 * SECONDS_IN_MINUTE
    )
    downsampled_first = get_artifact_dataframe(
        run=run,
        artifact_or_name=preprocessed_data_artifact_name,
        pkl_filename=os.path.join("not_windowed", "traditional_preprocessed_data.pkl"),
    )
    downsampled_last = get_artifact_dataframe(
        run=run,
        artifact_or_name=preprocessed_data_artifact_name,
        pkl_filename=os.path.join("not_windowed", "intermediate_preprocessed_data.pkl"),
    )

    datasets_to_plot = [
        (normalize_windows(raw_data), {"color": "k", "label": "original signal"}),
        (
            normalize_windows(downsampled_first),
            {"color": "y", "label": "downsampled first"},
        ),
        (
            normalize_windows(downsampled_last),
            {"color": "b", "label": "downsampled last"},
        ),
    ]
    windows_to_plot = raw_data.columns[0:1]
    plot_examples(datasets_to_plot=datasets_to_plot, windows_to_plot=windows_to_plot)
