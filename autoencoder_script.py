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
from models.denoising_autoencoder import MLPDenoisingAutoEncoder, create_autoencoder
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

# autoencoder = MLPDenoisingAutoEncoder(config=config)
# autoencoder.compile(
#     optimizer=config.optimizer, loss=config.loss, metrics=[config.metric]
# )

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
loaded_autoencoder = tf.keras.models.load_model(
    "/Users/williamdavies/Downloads/model-best (1).h5"
)
loaded_autoencoder.summary()

# %%
tf.keras.utils.plot_model(loaded_autoencoder, "model_plot1.png", show_shapes=True)
# %%
class MLPDenoisingAutoEncoder(Model):
    def __init__(self, config):
        super(MLPDenoisingAutoEncoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Dense(
                    config.encoder_1,
                    activation=config.encoder_activation_1,
                    input_shape=(config.timeseries_length,),
                ),
                layers.Dense(config.encoder_2, activation=config.encoder_activation_2),
                layers.Dense(config.encoder_3, activation=config.encoder_activation_3),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(
                    config.decoder_1,
                    activation=config.decoder_activation_1,
                    input_shape=(config.encoder_3,),
                ),
                layers.Dense(config.decoder_2, activation=config.decoder_activation_2),
                layers.Dense(config.decoder_3, activation=config.decoder_activation_3),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = MLPDenoisingAutoEncoder(config=config)
autoencoder.compile(
    optimizer=config.optimizer, loss=config.loss, metrics=[config.metric]
)

# %%
# need to call it on a batch before load_weights
autoencoder(train_data.values)

# %%
autoencoder.load_weights("/Users/williamdavies/Downloads/model-best.h5")

# %%
datestring = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_save_fp = os.path.join("models", "checkpoints", datestring)
# model_save_fp = os.path.join('models', 'checkpoints')
autoencoder.save_weights(model_save_fp)

# %%
loss_save_fp = os.path.join("models", f"{datestring}_loss.txt")
with open(loss_save_fp, "w") as f:
    for loss in history.history["loss"]:
        f.write(str(loss) + "\n")

# %%
val_loss_save_fp = os.path.join("models", f"{datestring}_val_loss.txt")
with open(val_loss_save_fp, "w") as f:
    for loss in history.history["val_loss"]:
        f.write(str(loss) + "\n")

# %%
epoch_save_fp = os.path.join("models", f"{datestring}_epoch.txt")
with open(epoch_save_fp, "w") as f:
    for epoch in history.epoch:
        f.write(str(epoch) + "\n")

# %%
loss = []
with open(loss_save_fp, "r") as f:
    for line in f:
        loss.append(float(line.strip()))
# %%
loaded_autoencoder = MLPDenoisingAutoEncoder(timeseries_length=timeseries_length)
loaded_autoencoder.compile(optimizer="adam", loss="mae")
loaded_autoencoder.load_weights(model_save_fp)
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

# %%
timeseries_length = len(data)
untrained_autoencoder = MLPDenoisingAutoEncoder(timeseries_length=timeseries_length)
untrained_autoencoder.compile(optimizer="adam", loss="mae")
