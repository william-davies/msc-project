import os
import pandas as pd
import wandb
import numpy as np

#%%
from msc_project.constants import DENOISING_AUTOENCODER_PROJECT_NAME, BASE_DIR

# run_dir = "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/results/hrv/unique-dream-323"
# inf_raw_data_heartpy_output = pd.read_pickle(os.path.join(run_dir, "inf_raw_data_hrv.pkl"))
# empatica_raw_data_heartpy_output = pd.read_pickle(os.path.join(run_dir, "empatica_raw_data_hrv.pkl"))
from msc_project.scripts.hrv.get_rmse import get_rmse

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


def read_hrv_of_interest(filename) -> pd.DataFrame:
    return replace_masked_with_nan(pd.read_pickle(os.path.join(run_dir, filename)))


def replace_masked_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    cond = df.applymap(lambda element: element is np.ma.masked)
    replaced = df.mask(cond=cond)
    return replaced


if __name__ == "__main__":
    run_dir = "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/results/hrv_rmse/spring-sunset-337"

    inf_raw = read_hrv_of_interest("inf_raw_data_hrv_of_interest.pkl")
    emp_raw = read_hrv_of_interest("empatica_raw_data_hrv_of_interest.pkl")
    emp_traditional = read_hrv_of_interest(
        "empatica_traditional_preprocessed_data_hrv_of_interest.pkl"
    )
    emp_intermediate = read_hrv_of_interest(
        "empatica_intermediate_preprocessed_data_hrv_of_interest.pkl"
    )
    emp_proposed = read_hrv_of_interest(
        "empatica_proposed_denoised_data_hrv_of_interest.pkl"
    )

    rmse = get_rmse(gt_hrv_metrics=inf_raw, other_hrv_metrics=emp_raw)
