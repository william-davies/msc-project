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

    all_rmses = pd.read_pickle(os.path.join(run_dir, "all_rmses.pkl"))
    normalized = normalize_rmse(all_rmses)

    normalized.plot.bar()
    plt.show()

    all_rmses.loc["ibi"].plot.bar()
    plt.show()

    for index, row in all_rmses.iterrows():
        row.plot.bar()
        plt.title(row.name)
        plt.tight_layout()
        plt.show()

    rmse_plots_dir = os.path.join(run_dir, "rmse_plots", "include_raw")
    for index, row in all_rmses.iterrows():
        row = row[:]
        row.plot.bar()
        plt.title(row.name)
        plt.ylabel("rmse")

        for index, value in enumerate(row):
            plt.text(index, value, "{0:.1f}".format(value))

        plt.tight_layout()
        plt.savefig(os.path.join(rmse_plots_dir, f"{row.name}.png"))
        plt.clf()
