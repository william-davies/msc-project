"""
Inspect how the merged signal arises (by mean-ing) from the reconstructed windows.
"""
import os

import wandb
from matplotlib import pyplot as plt

from msc_project.constants import (
    DENOISING_AUTOENCODER_PROJECT_NAME,
    SheetNames,
    SECONDS_IN_MINUTE,
)
from msc_project.scripts.evaluate_autoencoder import (
    download_artifact_model,
    get_reconstructed_df,
)
from msc_project.scripts.hrv.merge_denoised_windows import (
    convert_relative_timedelta_to_absolute_timedelta,
)
from msc_project.scripts.plot_raw_data import plot_signal
from msc_project.scripts.utils import get_artifact_dataframe

if __name__ == "__main__":
    model_artifact_name = "trained_on_EmLBVP:v0"
    sheet_name = SheetNames.EMPATICA_LEFT_BVP.value
    data_name: str = "only_downsampled"
    config = {"data_name": data_name}

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type="inspect_merged_signal",
        save_code=True,
        config=config,
    )

    # 1. get reconstructed windows
    only_downsampled_data = get_artifact_dataframe(
        run=run,
        artifact_or_name=f"{sheet_name}_preprocessed_data:v5",
        pkl_filename=os.path.join("windowed", f"{data_name}_data.pkl"),
    )

    autoencoder = download_artifact_model(run=run, artifact_or_name=model_artifact_name)
    reconstructed_windows = get_reconstructed_df(
        to_reconstruct=only_downsampled_data.T,
        autoencoder=autoencoder,
    ).T
    reconstructed_windows = convert_relative_timedelta_to_absolute_timedelta(
        reconstructed_windows
    )

    # 2. load empatica merged signal
    empatica_merged_signal_artifact_name: str = "EmLBVP_merged_signal:v1"
    merged_signal = get_artifact_dataframe(
        run=run,
        artifact_or_name=empatica_merged_signal_artifact_name,
        pkl_filename="merged_signal.pkl",
    )

    # 3. plot signals and compare
    arbitrary_example = merged_signal.columns[2]
    treatment_windows = reconstructed_windows[arbitrary_example]
    treatment_windows = (
        treatment_windows + 1
    )  # so that they're not all overlapping and illegible
    plt.figure()
    for idx in treatment_windows.iloc[:, :]:
        window = treatment_windows[idx]
        # this is very opaque code. basically plots all the reconstructed windows in different locations so that
        # you can actually look at them
        # %8 makes this periodic
        offset = ((idx - SECONDS_IN_MINUTE) % 8) / 1.5
        plot_signal(signal=window + offset)
    plot_signal(signal=merged_signal[arbitrary_example])
    plt.title(arbitrary_example)
    plt.show()
