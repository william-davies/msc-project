import csv
import datetime
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb
from tensorflow import keras

from constants import (
    PARTICIPANT_DIRNAMES_WITH_EXCEL,
    PARTICIPANT_NUMBER_PATTERN,
)
from models.denoising_autoencoder import create_autoencoder
from utils import read_dataset_csv
from wandb.keras import WandbCallback
from tensorflow.keras.models import Model
from tensorflow.keras import layers

import tensorflow as tf

# %%

random_state = np.random.RandomState(42)
NUM_PARTICIPANTS = len(PARTICIPANT_DIRNAMES_WITH_EXCEL)
validation_size = round(NUM_PARTICIPANTS * 0.3)

data = read_dataset_csv(
    "Stress Dataset/preprocessed_data/downsampled16Hz_10sec_window_5sec_overlap.csv"
)


def get_participant_number(string):
    """

    :param string:
    :return: int:
    """
    participant_number = PARTICIPANT_NUMBER_PATTERN.search(string).group(1)
    return participant_number


PARTICPANT_NUMBERS_WITH_EXCEL = list(
    map(get_participant_number, PARTICIPANT_DIRNAMES_WITH_EXCEL)
)

validation_participants = random_state.choice(
    a=PARTICPANT_NUMBERS_WITH_EXCEL, size=validation_size, replace=False
)
validation_participants = set(validation_participants)


def get_train_val_columns(data, validation_participants):
    """
    Get DataFrame columns that correspond to participants in training set and validation set.
    :param data: pd.DataFrame:
    :return:
    """
    number_pattern = re.compile("^P(\d{1,2})_")

    train_columns = []
    val_columns = []
    for participant_column in data.columns:
        participant_number = number_pattern.match(participant_column).group(1)
        if participant_number in validation_participants:
            val_columns.append(participant_column)
        else:
            train_columns.append(participant_column)
    return train_columns, val_columns


train_columns, val_columns = get_train_val_columns(data, validation_participants)

train_data = data.filter(items=train_columns).T
val_data = data.filter(items=val_columns).T

timeseries_length = len(data)


# %%
# autoencoder.build(train_data.shape)
# autoencoder.summary()


# %%
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=1e-2,
    patience=200,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)


# %%

bottleneck_size = 8
# Start a run, tracking hyperparameters
wandb.init(
    project="denoising-autoencoder",
    # Set entity to specify your username or team name
    # ex: entity="carey",
    config={
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
        "metric": "accuracy",
        "epoch": 6,
        "batch_size": 32,
        "timeseries_length": timeseries_length,
    },
)
config = wandb.config

autoencoder = create_autoencoder(config)

# %%
wandbcallback = WandbCallback(save_weights_only=False)
history = autoencoder.fit(
    train_data,
    train_data,
    epochs=config.epoch,
    batch_size=config.batch_size,
    validation_data=(val_data, val_data),
    callbacks=[wandbcallback],
    shuffle=True,
)

wandb.finish()

# %%
api = wandb.Api()

# run is specified by <entity>/<project>/<run id>
run = api.run("william-davies/denoising-autoencoder/2c7vl5qp")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
metrics_dataframe.to_csv("metrics.csv")

# %%
loaded_autoencoder = tf.keras.models.load_model(
    "/Users/williamdavies/Downloads/model-best (1).h5"
)
loaded_autoencoder.summary()

# %%
tf.keras.utils.plot_model(loaded_autoencoder, "model_plot1.png", show_shapes=True)


# %%
plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"][1:], label="Training Loss")
plt.plot(history.history["val_loss"][1:], label="Validation Loss")
plt.legend()
plt.show()

# %%
example = train_data.T.iloc[0].values

# %%
decoded_train_examples = tf.stop_gradient(autoencoder.call(train_data))

# %%
plt.plot(normalised_train_data[16], "b")
plt.plot(decoded_train_examples[16], "r")
plt.show()

# %%
decoded_val_examples = tf.stop_gradient(autoencoder.call(normalised_val_data))

# %%
plt.figure(figsize=(120, 20))
plt.plot(normalised_val_data[16], "b")
plt.plot(decoded_val_examples[16], "r")
plt.show()

# %%
delta = normalised_train_data[10] - decoded_examples[10]

# %%
decoded_all_examples = tf.stop_gradient(autoencoder(data.values.T))
decoded_train_examples = tf.stop_gradient(autoencoder(train_data.values))
decoded_val_examples = tf.stop_gradient(autoencoder(val_data.values))

# %%
example_idx = 300
plt.figure(figsize=(8, 6))
plt.title("Train example")
plt.plot(train_data.values[example_idx], "b")
plt.plot(decoded_train_examples[example_idx], "r")
plt.show()

# %%
example_idx = 200
plt.figure(figsize=(8, 6))
plt.title("Validation example")
plt.plot(val_data.values[example_idx], "b")
plt.plot(decoded_val_examples[example_idx], "r")
plt.show()

# %%
# plt.figure(figsize=(120, 20))
example_idx = 1000
plt.figure(figsize=(8, 6))
plt.plot(data.values.T[example_idx], "b")
plt.plot(decoded_all_examples[example_idx], "r")
plt.show()

# %%
plt.figure(figsize=(120, 20))
save_filepath = "autoencoder_plots/{}.png"

for idx in range(len(normalised_data)):
    title = data.columns[idx]
    plt.title(title)

    plt.plot(normalised_data[idx], "b")
    plt.plot(decoded_all_examples[idx], "r")

    plt.xlabel("Frames")
    plt.ylabel("BVP")

    plt.savefig(save_filepath.format(title), format="png")
    plt.clf()

# %%

plt.figure(figsize=(120, 20))
plt.plot(decoded_all_examples[0], "b")
plt.plot(decoded_all_examples[1], "r")
plt.show()
