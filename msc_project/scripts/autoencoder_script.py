import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import wandb

from msc_project.constants import (
    PARTICIPANT_DIRNAMES_WITH_EXCEL,
    PARTICPANT_NUMBERS_WITH_EXCEL,
    BASE_DIR,
)
from msc_project.models.denoising_autoencoder import create_autoencoder

from utils import read_dataset_csv
from wandb.keras import WandbCallback

import tensorflow as tf

# %%
class DatasetPreparer:
    """
    Reads preprocessed signal data. Splits it into train, val, noisy.
    """

    def __init__(self, data_dirname, noisy_tolerance):
        self.data_dirname = data_dirname
        self.noisy_tolerance = noisy_tolerance

    def get_dataset(self):
        signals_fp = os.path.join(
            BASE_DIR,
            "data",
            "Stress Dataset",
            "preprocessed_data",
            self.data_dirname,
            "signal.csv",
        )
        signals = read_dataset_csv(signals_fp)
        clean_signals, noisy_signals = self.filter_noisy_signals(signals=signals)

        validation_participants = self.get_validation_participants()

        train_columns, val_columns = self.get_train_val_columns(
            clean_signals, validation_participants
        )

        train_signals = signals.filter(items=train_columns).T
        val_signals = signals.filter(items=val_columns).T

        return train_signals, val_signals, noisy_signals.T

    def get_validation_participants(self):
        """

        :return: int[]: validation participant numbers
        """
        random_state = np.random.RandomState(42)
        NUM_PARTICIPANTS = len(PARTICIPANT_DIRNAMES_WITH_EXCEL)
        validation_size = round(NUM_PARTICIPANTS * 0.3)
        validation_participants = random_state.choice(
            a=PARTICPANT_NUMBERS_WITH_EXCEL, size=validation_size, replace=False
        )
        validation_participants = set(validation_participants)
        return validation_participants

    def filter_noisy_signals(self, signals):
        """
        Split signals into 2 DataFrame. 1 is clean signals. 1 is noisy (as determined by self.noisy_tolerance) signals.
        :return:
        """

        def filter_noisy_signal_keys():
            noisy_frame_proportions_fp = os.path.join(
                self.data_dirname, "noisy_frame_proportions.json"
            )
            with open(noisy_frame_proportions_fp, "r") as fp:
                noisy_frame_proportions = json.load(fp)

            clean_keys = []
            noisy_keys = []

            for key, noisy_proportion in noisy_frame_proportions.items():
                if noisy_proportion <= self.noisy_tolerance:
                    clean_keys.append(key)
                else:
                    noisy_keys.append(key)

            return clean_keys, noisy_keys

        clean_columns, noisy_columns = filter_noisy_signal_keys()
        clean_signals = signals.filter(clean_columns)
        noisy_signals = signals.filter(noisy_columns)

        return clean_signals, noisy_signals

    def get_train_val_columns(self, signals, validation_participants):
        """
        Get DataFrame columns that correspond to participants in training set and validation set.
        :param signals: pd.DataFrame:
        :return:
        """
        participant_number_pattern = re.compile("^P(\d{1,2})_")

        train_columns = []
        val_columns = []
        for participant_column in signals.columns:
            participant_number = participant_number_pattern.match(
                participant_column
            ).group(1)
            if participant_number in validation_participants:
                val_columns.append(participant_column)
            else:
                train_columns.append(participant_column)
        return train_columns, val_columns


dataset_preparer = DatasetPreparer(
    data_dirname="/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/data/preprocessed_data/noisy_labelled",
    noisy_tolerance=0,
)
train_signals, val_signals, noisy_signals = dataset_preparer.get_dataset()


# %%
def train_autoencoder(resume, train_signals, val_signals, run_id=None):
    """

    :param resume: bool: resume training a previous model
    :return:
    """
    if resume != bool(run_id):
        raise ValueError

    project_name = "denoising-autoencoder"
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-2,
        patience=200,
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
        wandb.init(
            id=run_id,
            project=project_name,
            resume="must",
            config={**base_config, "epoch": 10},
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
            callbacks=[wandbcallback],
            shuffle=True,
            initial_epoch=wandb.run.step,
        )

    else:
        wandb.init(
            project=project_name,
            config={**base_config, "epoch": 3000},
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
            callbacks=[wandbcallback],
            shuffle=True,
        )
    wandb.finish()
    return autoencoder, history


# %%
# autoencoder, history = train_autoencoder(resume=True, train_signals=train_signals, val_signals=val_signals, run_id='3f1ei8kq')
autoencoder, history = train_autoencoder(
    resume=False, train_signals=train_signals, val_signals=val_signals
)
