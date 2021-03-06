import os
from datetime import datetime

import heartpy as hp
import pandas as pd
import wandb

from msc_project.constants import DENOISING_AUTOENCODER_PROJECT_NAME, BASE_DIR
from msc_project.scripts.evaluate_autoencoder import (
    download_artifact_model,
    get_reconstructed_df,
)
from msc_project.scripts.get_preprocessed_data import get_freq

# %%
from msc_project.scripts.utils import get_artifact_dataframe


def hp_process_wrapper(hrdata, sample_rate, report_time, calc_freq):
    """
    rows is time index. wrapper for hp.process. handles BadSignalWarning.
    :param hrdata:
    :param sample_rate:
    :param report_time:
    :param calc_freq:
    :return:
    """
    try:
        wd, m = hp.process(
            hrdata,
            sample_rate=sample_rate,
            report_time=report_time,
            calc_freq=calc_freq,
        )
        merged = {**wd, **m}
        merged["warning"] = ""
    except hp.exceptions.BadSignalWarning as e:
        print(f"series name: {hrdata.name}")
        print(f"exception: {e}")
        return None
    except UserWarning as w:
        print(f"series name: {hrdata.name}")
        print(f"warning: {w}")
        merged = {**wd, **m}
        merged["warning"] = w.args[0]
    return pd.Series(merged)


def get_hrv(signal_data: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function that gets HRV metrics for `signal_data`.
    :param signal_data:
    :return:
    """
    fs = get_freq(signal_data.index)
    hrv = signal_data.apply(
        func=hp_process_wrapper,
        axis=0,
        sample_rate=fs,
        report_time=False,
        calc_freq=True,
    )
    return hrv


if __name__ == "__main__":
    model_artifact_name = "trained_on_EmLBVP:v0"
    upload_artifact = True

    run = wandb.init(
        project=DENOISING_AUTOENCODER_PROJECT_NAME,
        job_type="hrv_evaluation",
        save_code=True,
    )

    inf_raw_data = get_artifact_dataframe(
        run=run,
        artifact_or_name="Inf_preprocessed_data:v4",
        pkl_filename="windowed_raw_data.pkl",
    )
    empatica_raw_data = get_artifact_dataframe(
        run=run,
        artifact_or_name="EmLBVP_preprocessed_data:v4",
        pkl_filename="windowed_raw_data.pkl",
    )
    empatica_intermediate_preprocessed_data = get_artifact_dataframe(
        run=run,
        artifact_or_name="EmLBVP_preprocessed_data:v4",
        pkl_filename="windowed_intermediate_preprocessed_data.pkl",
    )
    empatica_only_downsampled_data = get_artifact_dataframe(
        run=run,
        artifact_or_name="EmLBVP_preprocessed_data:v4",
        pkl_filename="windowed_only_downsampled_data.pkl",
    )

    autoencoder = download_artifact_model(run=run, artifact_or_name=model_artifact_name)
    empatica_reconstructed_data = get_reconstructed_df(
        to_reconstruct=empatica_only_downsampled_data.T,
        autoencoder=autoencoder,
    ).T

    run_dir = os.path.join(BASE_DIR, "results", "hrv", run.name)
    os.makedirs(run_dir)

    print(f"inf_raw_data_hrv start: {datetime.now()}")
    inf_raw_data_hrv = get_hrv(signal_data=inf_raw_data)
    inf_raw_data_hrv.to_pickle(os.path.join(run_dir, "inf_raw_data_hrv.pkl"))
    print(f"inf_raw_data_hrv end: {datetime.now()}")

    print(f"empatica_raw_data_hrv start: {datetime.now()}")
    empatica_raw_data_hrv = get_hrv(signal_data=empatica_raw_data)
    empatica_raw_data_hrv.to_pickle(os.path.join(run_dir, "empatica_raw_data_hrv.pkl"))
    print(f"empatica_raw_data_hrv end: {datetime.now()}")

    print(f"empatica_only_downsampled_data_hrv start: {datetime.now()}")
    empatica_only_downsampled_data_hrv = get_hrv(
        signal_data=empatica_only_downsampled_data
    )
    empatica_only_downsampled_data_hrv.to_pickle(
        os.path.join(run_dir, "empatica_only_downsampled_data_hrv.pkl")
    )
    print(f"empatica_only_downsampled_data_hrv end: {datetime.now()}")

    print(f"empatica_intermediate_preprocessed_data_hrv start: {datetime.now()}")
    empatica_intermediate_preprocessed_data_hrv = get_hrv(
        signal_data=empatica_intermediate_preprocessed_data
    )
    empatica_intermediate_preprocessed_data_hrv.to_pickle(
        os.path.join(run_dir, "empatica_intermediate_preprocessed_data_hrv.pkl")
    )
    print(f"empatica_intermediate_preprocessed_data_hrv end: {datetime.now()}")

    print(f"empatica_reconstructed_data_hrv start: {datetime.now()}")
    empatica_reconstructed_data_hrv = get_hrv(signal_data=empatica_reconstructed_data)
    empatica_reconstructed_data_hrv.to_pickle(
        os.path.join(run_dir, "empatica_reconstructed_data_hrv.pkl")
    )
    print(f"empatica_reconstructed_data_hrv end: {datetime.now()}")

    # guard to save wandb storage
    if upload_artifact:
        hrv_artifact = wandb.Artifact(name="get_hrv", type="get_hrv")
        hrv_artifact.add_dir(run_dir)
        run.log_artifact(hrv_artifact)
    run.finish()
