"""
Inspect how the merged signal arises (by mean-ing) from the reconstructed windows.
Also compare merged denoised signal to model input.
"""
import os

import wandb
from matplotlib import pyplot as plt

from msc_project.constants import (
    DENOISING_AUTOENCODER_PROJECT_NAME,
    SheetNames,
    SECONDS_IN_MINUTE,
)
from msc_project.scripts.data_processing.denoise_and_merge_entire_signal import (
    convert_relative_timedelta_to_absolute_timedelta,
)
from msc_project.scripts.data_processing.get_preprocessed_data import normalize_windows
from msc_project.scripts.evaluate_autoencoder import (
    download_artifact_model,
    get_reconstructed_df,
)
from msc_project.scripts.hrv.get_rmse import get_noisy_proportions

from msc_project.scripts.plot_raw_data import plot_signal
from msc_project.scripts.utils import get_committed_artifact_dataframe

if __name__ == "__main__":
    preprocessed_data_artifact_name: str = "EmLBVP_preprocessed_data:v8"
    model_artifact_name = "trained_on_EmLBVP:v6"
    merged_signal_artifact_name: str = "EmLBVP_merged_signal:v3"
    data_name: str = "only_downsampled"
    config = {"data_name": data_name}

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type="inspect_merged_signal",
        save_code=True,
        config=config,
    )

    # 1. get reconstructed windows
    only_downsampled_data = get_committed_artifact_dataframe(
        run=run,
        artifact_or_name=preprocessed_data_artifact_name,
        pkl_filename=os.path.join("windowed", f"{data_name}_data.pkl"),
    )
    noisy_mask = get_committed_artifact_dataframe(
        run=run,
        artifact_or_name=preprocessed_data_artifact_name,
        pkl_filename=os.path.join("windowed", f"noisy_mask.pkl"),
    )
    noisy_proportions = get_noisy_proportions(noisy_mask=noisy_mask)
    noise_per_treatment = noisy_proportions.groupby(
        level=["participant", "treatment_label"]
    ).mean()

    autoencoder = download_artifact_model(run=run, artifact_or_name=model_artifact_name)
    reconstructed_windows = get_reconstructed_df(
        to_reconstruct=only_downsampled_data.T,
        autoencoder=autoencoder,
    ).T
    reconstructed_windows = convert_relative_timedelta_to_absolute_timedelta(
        reconstructed_windows
    )

    # 2. load merged signal
    merged_signal = get_committed_artifact_dataframe(
        run=run,
        artifact_or_name=merged_signal_artifact_name,
        pkl_filename="merged_signal.pkl",
    )

    # get model input
    not_windowed_downsampled_data = get_committed_artifact_dataframe(
        run=run,
        artifact_or_name=preprocessed_data_artifact_name,
        pkl_filename=os.path.join("not_windowed", f"{data_name}_data.pkl"),
    )

    # 3. plot signals and compare
    arbitrary_example = merged_signal.columns[0]
    treatment_windows = reconstructed_windows[arbitrary_example]
    treatment_windows = (
        treatment_windows + 1
    )  # so that they're not all overlapping and illegible
    plt.figure()
    scaling_factor = 4
    for idx in treatment_windows.iloc[:, :]:
        window = treatment_windows[idx]
        # this is very opaque code. basically plots all the reconstructed windows in different locations so that
        # you can actually look at them
        # %8 makes this periodic
        offset = ((idx - SECONDS_IN_MINUTE) % 8) / 1.5 + scaling_factor
        plot_signal(signal=window + offset)
    plot_signal(
        signal=scaling_factor * merged_signal[arbitrary_example],
        color="k",
        linestyle="--",
    )
    plt.title(arbitrary_example)
    plt.ylabel("PPG (a.u)")
    plt.xlabel("Time in session (s)")
    ax = plt.gca()
    ax.axes.yaxis.set_ticks([])
    plt.show()

    plt.figure()
    plot_signal(
        signal=normalize_windows(not_windowed_downsampled_data[arbitrary_example]),
        label="just downsampled",
    )
    plot_signal(
        signal=normalize_windows(merged_signal[arbitrary_example]), label="dae denoised"
    )
    plt.title(arbitrary_example)
    plt.legend()
    plt.show()
