"""
Denoised windows are say 8s long and overlap by say 1s. We need to merge these into a single coherent say 3min signal.
"""
import os

import pandas as pd
import wandb

from msc_project.constants import (
    DENOISING_AUTOENCODER_PROJECT_NAME,
    SECONDS_IN_MINUTE,
    SheetNames,
)
from msc_project.scripts.evaluate_autoencoder import (
    download_artifact_model,
    get_reconstructed_df,
)
from msc_project.scripts.get_preprocessed_data import get_freq
from msc_project.scripts.get_sheet_raw_data import get_timedelta_index
from msc_project.scripts.utils import get_artifact_dataframe


def merge_windows(reconstructed_windows):
    """
    Following Lee et al. 2019.
    :param reconstructed_windows:
    :return:
    """
    merged_reconstructed_signals = convert_relative_timedelta_to_absolute_timedelta(
        reconstructed_windows
    )

    meaned = merged_reconstructed_signals.groupby(
        level=["participant", "treatment_label", "series_label"], axis=1
    ).mean()
    return meaned


def convert_relative_timedelta_to_absolute_timedelta(
    relative_timedelta_windows: pd.DataFrame,
) -> pd.DataFrame:
    """
    0s - 8s -> 62s - 70s. Relative to start of window -> relative to start of treatment.
    :param relative_timedelta_windows:
    :return:
    """
    full_signal_index = get_timedelta_index(
        start_time=SECONDS_IN_MINUTE,
        end_time=4 * SECONDS_IN_MINUTE,
        frequency=get_freq(index=relative_timedelta_windows.index),
    )
    absolute_timedelta_windows = pd.DataFrame(
        data=float("nan"),
        columns=relative_timedelta_windows.columns,
        index=full_signal_index,
    )
    for window_idx in relative_timedelta_windows:
        window = relative_timedelta_windows[window_idx]
        window_start = window.name[-1]
        window_start = pd.to_timedelta(window_start, unit="second")
        absolute_timedelta_indexes = window_start + window.index
        absolute_timedelta_windows.loc[
            absolute_timedelta_indexes, window_idx
        ] = window.values

    return absolute_timedelta_windows


if __name__ == "__main__":
    model_artifact_name = "trained_on_Inf:v5"
    sheet_name = SheetNames.INFINITY.value
    preprocessed_data_artifact_version: int = 7
    data_name: str = "only_downsampled"
    config = {"data_name": data_name}
    upload_artifact: bool = True

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type="merge",
        save_code=True,
        config=config,
    )

    only_downsampled_data = get_artifact_dataframe(
        run=run,
        artifact_or_name=f"{sheet_name}_preprocessed_data:v{preprocessed_data_artifact_version}",
        pkl_filename=os.path.join("windowed", f"{data_name}_data.pkl"),
    )

    autoencoder = download_artifact_model(run=run, artifact_or_name=model_artifact_name)
    reconstructed_windows = get_reconstructed_df(
        to_reconstruct=only_downsampled_data.T,
        autoencoder=autoencoder,
    ).T

    meaned = merge_windows(reconstructed_windows=reconstructed_windows)

    if upload_artifact:
        merged_signal_artifact = wandb.Artifact(
            name=f"{sheet_name}_merged_signal", type="merged_signal", metadata=config
        )
        with merged_signal_artifact.new_file("merged_signal.pkl", "w") as f:
            meaned.to_pickle(f.name)
        run.log_artifact(merged_signal_artifact)
    run.finish()
