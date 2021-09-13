import os

import heartpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import wandb

from msc_project.constants import DENOISING_AUTOENCODER_PROJECT_NAME, BASE_DIR
from msc_project.scripts.evaluate_autoencoder import (
    download_artifact_if_not_already_downloaded,
    get_model,
    get_reconstructed_df,
)
from msc_project.scripts.get_preprocessed_data import get_freq
import tensorflow as tf

run_dir = "/results/hrv_rmse/spring-sunset-337"
rmse_dir = os.path.join(run_dir, "to_upload")


def get_rmse(filename: str) -> pd.DataFrame:
    return pd.read_pickle(os.path.join(rmse_dir, filename))


empatica_raw = get_rmse("empatica_raw_data_rmse.pkl")
empatica_traditional_preprocessed = get_rmse(
    "empatica_traditional_preprocessed_data_rmse.pkl"
)
empatica_intermediate_preprocessed = get_rmse(
    "empatica_intermediate_preprocessed_data_rmse.pkl"
)
empatica_proposed_denoised = get_rmse("empatica_proposed_denoised_data_rmse.pkl")

all_rmses = pd.concat(
    objs=[
        empatica_raw,
        empatica_traditional_preprocessed,
        empatica_intermediate_preprocessed,
        empatica_proposed_denoised,
    ],
    axis=1,
    keys=[
        "empatica_raw",
        "empatica_traditional_preprocessed",
        "empatica_intermediate_preprocessed",
        "empatica_proposed_denoised",
    ],
)
min = all_rmses.min(axis=1)
max = all_rmses.max(axis=1)
normalized = all_rmses.subtract(min, axis=0).divide(max - min, axis=0)

normalized.plot.bar()
plt.show()
