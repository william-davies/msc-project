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
# wandb.init(
#     project="denoising-autoencoder",
#     # Set entity to specify your username or team name
#     # ex: entity="carey",
#     config={
#         "encoder_1": bottleneck_size * 2 * 2,
#         "encoder_activation_1": "relu",
#         "encoder_2": bottleneck_size * 2,
#         "encoder_activation_2": "relu",
#         "encoder_3": bottleneck_size,
#         "encoder_activation_3": "relu",
#         "decoder_1": bottleneck_size * 2,
#         "decoder_activation_1": "relu",
#         "decoder_2": bottleneck_size * 2 * 2,
#         "decoder_activation_2": "relu",
#         "decoder_3": timeseries_length,
#         "decoder_activation_3": "sigmoid",
#         "optimizer": "adam",
#         "loss": "mae",
#         "metric": [None],
#         "epoch": 3000,
#         "batch_size": 32,
#         "timeseries_length": timeseries_length,
#     },
#     force=True,
#     allow_val_change=False
# )

wandb.init(
    id="ytenhze8",
    project="denoising-autoencoder",
    resume="must",
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
        "metric": [None],
        "epoch": 3000,
        "batch_size": 32,
        "timeseries_length": timeseries_length,
    },
    force=True,
    allow_val_change=False,
)

config = wandb.config

# %%
best_model = wandb.restore("model-best.h5")
autoencoder = tf.keras.models.load_model(best_model.name)
# %%

autoencoder = create_autoencoder(config)

# %%
wandbcallback = WandbCallback(save_weights_only=False, monitor="val_loss")
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
loaded_autoencoder = tf.keras.models.load_model(
    "/Users/williamdavies/Downloads/model-best.h5"
)
loaded_autoencoder.summary()

# %%
tf.keras.utils.plot_model(loaded_autoencoder, "model_plot.png", show_shapes=True)


# %%
plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"][1:], label="Training Loss")
plt.plot(history.history["val_loss"][1:], label="Validation Loss")
plt.legend()
plt.show()

# %%
example = train_data.T.iloc[0].values

# %%
autoencoder = loaded_autoencoder

# %%
decoded_all_examples = tf.stop_gradient(autoencoder(data.values.T))
decoded_train_examples = tf.stop_gradient(autoencoder(train_data.values))
decoded_val_examples = tf.stop_gradient(autoencoder(val_data.values))

# %%
def plot_examples(
    original_data,
    reconstructed_data,
    example_type,
    epoch,
    save_dir=None,
    num_examples=5,
):
    """

    :param original_data: pd.DataFrame:
    :param reconstructed_data: pd.DataFrame:
    :param example_type: str: Train/Validation
    :param epoch: int:
    :param save_dir: str:
    :param num_examples: int:
    :return:
    """
    example_idxs = random_state.choice(
        a=len(original_data), size=num_examples, replace=False
    )
    plt.figure(figsize=(8, 6))
    for example_idx in example_idxs:
        window_label = original_data.iloc[example_idx].name
        plt.title(
            f"{example_type} example\n{window_label}\nExample index: {example_idx}"
        )
        plt.plot(original_data.values[example_idx], "b", label="original")
        plt.plot(reconstructed_data[example_idx], "r", label="denoised")
        plt.legend()

        if save_dir:
            save_filepath = os.path.join(save_dir, f"epoch-{epoch}_{window_label}.png")
            plt.savefig(save_filepath, format="png")
            plt.clf()
        else:
            plt.show()


# %%
plot_examples(
    val_data,
    decoded_val_examples,
    example_type="Validation",
    save_dir="3000-epochs",
    epoch=3000,
)

# %%
plot_examples(
    train_data,
    decoded_train_examples,
    example_type="Train",
    save_dir="3000-epochs",
    epoch=3000,
)

# %%
example_idx = 0
original_data = val_data
reconstructed_data = decoded_val_examples
example_type = "Validation"
plt.figure(figsize=(8, 6))
window_label = original_data.iloc[example_idx].name
plt.title(f"{example_type} example\n{window_label}\nExample index: {example_idx}")
plt.plot(original_data.values[example_idx], "b", label="original")
plt.plot(reconstructed_data[example_idx], "r", label="denoised")
plt.legend()

save_filepath = os.path.join(
    wandb.run.dir, f"epoch-{wandb.run.step-1}_{window_label}.png"
)
plt.savefig(save_filepath, format="png")
plt.clf()

# %%
wandb.save(os.path.join(wandb.run.dir, "*epoch*"))

# %%
# plt.show()

wandb.log({"chart": plt}, step=2999)


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
