import json
import os
import re
import sys
import unittest
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import wandb

from msc_project.constants import (
    PARTICIPANT_DIRNAMES_WITH_EXCEL,
    PARTICIPANT_NUMBERS_WITH_EXCEL,
    BASE_DIR,
)
from msc_project.models.denoising_autoencoder import create_autoencoder

from wandb.keras import WandbCallback

import tensorflow as tf
import pandas as pd

# %%
class DatasetPreparer:
    """
    Reads preprocessed signal data. Splits it into train, val, noisy.
    """

    def __init__(self, noise_tolerance, signals, noisy_mask):
        self.noise_tolerance = noise_tolerance
        self.signals = signals
        self.noisy_mask = noisy_mask

    def get_dataset(self):
        self.signals = self.normalize_windows(self.signals)
        clean_signals, noisy_signals = self.split_into_clean_and_noisy()

        validation_participants = self.get_validation_participants()

        train_signals, val_signals = self.split_into_train_and_val(
            clean_signals, validation_participants
        )

        return train_signals, val_signals, noisy_signals

    def normalize_windows(self, signals):
        """
        Normalize each window.
        :param signals:
        :return:
        """
        min_vals = signals.min(axis=0)
        max_vals = signals.max(axis=0)
        normalized = (signals - min_vals) / (max_vals - min_vals)
        return normalized

    def get_validation_participants(self):
        """

        :return: validation participant dirnames
        """
        random_state = np.random.RandomState(42)
        NUM_PARTICIPANTS = len(PARTICIPANT_DIRNAMES_WITH_EXCEL)
        validation_size = round(NUM_PARTICIPANTS * 0.3)
        validation_participants = random_state.choice(
            a=PARTICIPANT_DIRNAMES_WITH_EXCEL, size=validation_size, replace=False
        )
        return validation_participants

    def split_into_clean_and_noisy(self):
        """
        Split signals into 2 DataFrame. 1 is clean signals. 1 is noisy (as determined by self.noisy_tolerance) signals.
        :return:
        """
        noisy_proportions = self.noisy_mask.sum(axis=0) / self.noisy_mask.shape[0]

        is_clean = noisy_proportions <= self.noise_tolerance
        clean_idxs = is_clean.index[is_clean]
        noisy_idxs = is_clean.index[~is_clean]

        clean_signals = self.signals[clean_idxs]
        noisy_signals = self.signals[noisy_idxs]

        return clean_signals, noisy_signals

    def split_into_train_and_val(
        self, signals: pd.DataFrame, validation_participants
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split signals into train and val.
        :param signals:
        :param validation_participants:
        :return:
        """
        train = signals.drop(columns=validation_participants, level="participant")
        val = signals[validation_participants]
        return train, val


# %%
def train_autoencoder(
    resume: bool, train_signals, val_signals, epoch: int, run_id: str = ""
):
    """

    :param resume: resume training a previous model
    :param train_signals:
    :param val_signals:
    :param epoch: how many more epochs to train.
    :param run_id: wandb run id
    :return:
    """
    # you must have both or neither
    if resume != bool(run_id):
        raise ValueError

    project_name = "denoising-autoencoder"
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-3,
        patience=1000,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    timeseries_length = train_signals.shape[1]
    bottleneck_size = 8
    base_config = {
        "encoder_1": bottleneck_size * 2 * 2,
        "encoder_activation_1": "relu",
        "encoder_2": bottleneck_size * 2,
        "encoder_activation_2": "relu",
        "encoder_3": bottleneck_size,
        "encoder_activation_3": "relu",
        "decoder_1": bottleneck_size * 2,
        "decoder_activation_1": "relu",
        "decoder_2": bottleneck_size * 2 * 2,
        "decoder_activation_2": "relu",
        "decoder_3": timeseries_length,
        "decoder_activation_3": "sigmoid",
        "optimizer": "adam",
        "loss": "mae",
        "metric": [None],
        "batch_size": 32,
        "timeseries_length": timeseries_length,
    }

    if resume:

        wandb_summary = json.loads(
            wandb.restore(
                "wandb-summary.json", run_path=f"{project_name}/{run_id}"
            ).read()
        )

        wandb.init(
            id=run_id,
            project=project_name,
            resume="must",
            config={**base_config, "epoch": 12627 + epoch},
            force=True,
            allow_val_change=True,
        )
        best_model = wandb.restore("model-best.h5", run_path=f"{project_name}/{run_id}")
        autoencoder = tf.keras.models.load_model(best_model.name)
        wandbcallback = WandbCallback(save_weights_only=False, monitor="val_loss")

        history = autoencoder.fit(
            train_signals,
            train_signals,
            epochs=wandb.config.epoch,
            batch_size=wandb.config.batch_size,
            validation_data=(val_signals, val_signals),
            callbacks=[wandbcallback, early_stop],
            shuffle=True,
            initial_epoch=wandb.run.step,
        )

    else:
        wandb.init(
            project=project_name,
            config={**base_config, "epoch": epoch},
            force=True,
            allow_val_change=False,
        )
        autoencoder = create_autoencoder(wandb.config)
        wandbcallback = WandbCallback(save_weights_only=False, monitor="val_loss")

        history = autoencoder.fit(
            train_signals,
            train_signals,
            epochs=wandb.config.epoch,
            batch_size=wandb.config.batch_size,
            validation_data=(val_signals, val_signals),
            callbacks=[wandbcallback, early_stop],
            shuffle=True,
        )
    wandb.finish()
    return autoencoder, history


# %%
if __name__ == "__main__":
    dataset_preparer = DatasetPreparer(
        noise_tolerance=0,
        signals=pd.read_pickle(
            "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/data/Stress Dataset/dataframes/windowed_data_window_start.pkl"
        ),
        noisy_mask=pd.read_pickle(
            "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/data/Stress Dataset/dataframes/windowed_noisy_mask_window_start.pkl"
        ),
    )
    train_signals, val_signals, noisy_signals = dataset_preparer.get_dataset()

    # autoencoder, history = train_autoencoder(
    #     resume=True,
    #     train_signals=train_signals,
    #     val_signals=val_signals,
    #     run_id="8doiurqf",
    #     epoch=12999 - 12627,
    # )

    autoencoder, history = train_autoencoder(
        resume=False,
        train_signals=train_signals.T,
        val_signals=val_signals.T,
        epoch=2000,
    )
