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


def normalize_rmse(raw_rmse):
    min = raw_rmse.min(axis=1)
    max = raw_rmse.max(axis=1)
    normalized = raw_rmse.subtract(min, axis=0).divide(max - min, axis=0)
    return normalized


if __name__ == "__main__":
    run_dir = "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/results/hrv_rmse/spring-sunset-337"

    all_rmses = pd.read_pickle(
        os.path.join(run_dir, "only_clean_inf_windows", "all_rmses.pkl")
    )

    rmse_plots_dir = os.path.join(
        run_dir, "only_clean_inf_windows", "rmse_plots", "exclude_raw"
    )
    # os.makedirs(rmse_plots_dir)
    for index, row in all_rmses.iterrows():
        row = row[1:]
        row.plot.bar()
        plt.title(row.name)
        plt.ylabel("rmse")

        for index, value in enumerate(row):
            plt.text(index, value, "{0:.3g}".format(value))

        plt.tight_layout()
        plt.savefig(os.path.join(rmse_plots_dir, f"{row.name}.png"))
        plt.clf()
