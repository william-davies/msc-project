import pandas as pd
import wandb

from msc_project.constants import DENOISING_AUTOENCODER_PROJECT_NAME, SECONDS_IN_MINUTE
from msc_project.scripts.evaluate_autoencoder import (
    download_artifact_model,
    get_reconstructed_df,
)
from msc_project.scripts.get_preprocessed_data import get_freq
from msc_project.scripts.get_sheet_raw_data import get_timedelta_index
from msc_project.scripts.utils import get_artifact_dataframe


if __name__ == "__main__":
    model_artifact_name = "trained_on_EmLBVP:v0"

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type="merge",
        save_code=True,
    )

    empatica_only_downsampled_data = get_artifact_dataframe(
        run=run,
        artifact_or_name="EmLBVP_preprocessed_data:v4",
        pkl_filename="windowed_only_downsampled_data.pkl",
    )

    autoencoder = download_artifact_model(run=run, artifact_or_name=model_artifact_name)
    reconstructed_windows = get_reconstructed_df(
        to_reconstruct=empatica_only_downsampled_data.T,
        autoencoder=autoencoder,
    ).T

    full_signal_index = get_timedelta_index(
        start_time=SECONDS_IN_MINUTE,
        end_time=4 * SECONDS_IN_MINUTE,
        frequency=get_freq(index=reconstructed_windows.index),
    )
    merged_reconstructed_signals = pd.DataFrame(
        data=float("nan"),
        columns=reconstructed_windows.columns,
        index=full_signal_index,
    )

    for window_idx in reconstructed_windows:
        reconstructed_window = reconstructed_windows[window_idx]
        window_start = reconstructed_window.name[-1]
        window_start = pd.to_timedelta(window_start, unit="second")
        merged_signal_timedelta_indexes = window_start + reconstructed_window.index
        merged_reconstructed_signals.loc[
            merged_signal_timedelta_indexes, window_idx
        ] = reconstructed_window.values

    meaned = merged_reconstructed_signals.groupby(
        level=["participant", "treatment_label"], axis=1
    ).mean()
