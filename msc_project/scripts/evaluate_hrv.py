import os
import pandas as pd
import wandb

#%%
from msc_project.constants import DENOISING_AUTOENCODER_PROJECT_NAME, BASE_DIR
from msc_project.scripts.get_hrv import get_artifact_dataframe

# run_dir = "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/results/hrv/unique-dream-323"
# inf_raw_data_heartpy_output = pd.read_pickle(os.path.join(run_dir, "inf_raw_data_hrv.pkl"))
# empatica_raw_data_heartpy_output = pd.read_pickle(os.path.join(run_dir, "empatica_raw_data_hrv.pkl"))

metrics_of_interest = [
    "bpm",
    "ibi",
    "sdnn",
    "rmssd",
    "pnn20",
    "pnn50",
    "hf",
    "sdsd",
    "hr_mad",
]


def get_hrv_metrics(heartpy_output: pd.DataFrame) -> pd.DataFrame:
    return heartpy_output.loc[metrics_of_interest]


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


if __name__ == "__main__":
    get_hrv_version: int = 3
    upload_artifacts: bool = True

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

    inf_raw_data_hrv = get_hrv_metrics(inf_raw_data_heartpy_output)
    empatica_raw_data_hrv = get_hrv_metrics(empatica_raw_data_heartpy_output)
    empatica_traditional_preprocessed_data_hrv = get_hrv_metrics(
        empatica_traditional_preprocessed_data_heartpy_output
    )
    empatica_intermediate_preprocessed_data_hrv = get_hrv_metrics(
        empatica_intermediate_preprocessed_data_heartpy_output
    )
    empatica_proposed_denoised_data_hrv = get_hrv_metrics(
        empatica_proposed_denoised_data_heartpy_output
    )

    run_dir = os.path.join(BASE_DIR, "results", "hrv_rmse", run.name)
    os.makedirs(run_dir)

    rmse_info = [
        (empatica_raw_data_hrv, "empatica_raw_data"),
        (
            empatica_traditional_preprocessed_data_hrv,
            "empatica_traditional_preprocessed_data",
        ),
        (
            empatica_intermediate_preprocessed_data_hrv,
            "empatica_intermediate_preprocessed_data",
        ),
        (empatica_proposed_denoised_data_hrv, "empatica_proposed_denoised_data"),
    ]

    for (hrv_data, data_name) in rmse_info:
        rmse = get_rmse(gt_hrv_metrics=inf_raw_data_hrv, other_hrv_metrics=hrv_data)
        rmse.to_pickle(os.path.join(run_dir, f"{data_name}_rmse.pkl"))

    if upload_artifacts:
        artifact = wandb.Artifact(name="hrv_rmse", type="hrv")
        artifact.add_dir(run_dir)
        run.log_artifact(artifact)
    run.finish()
