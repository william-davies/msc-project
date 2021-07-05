import csv
import datetime
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow import keras

from constants import (
    PARTICIPANT_DIRNAMES_WITH_EXCEL,
    PARTICIPANT_NUMBER_PATTERN,
)
from models.denoising_autoencoder import MLPDenoisingAutoEncoder
from utils import read_dataset_csv

# %%
import tensorflow as tf

# %%

random_state = np.random.RandomState(42)
NUM_PARTICIPANTS = len(PARTICIPANT_DIRNAMES_WITH_EXCEL)
validation_size = round(NUM_PARTICIPANTS * 0.3)

# %%
data = read_dataset_csv(
    "Stress Dataset/preprocessed_data/downsampled16Hz_10sec_window_5sec_overlap.csv"
)

# %%


def get_participant_number(string):
    participant_number = PARTICIPANT_NUMBER_PATTERN.search(string).group(1)
    return participant_number


PARTICPANT_NUMBERS_WITH_EXCEL = list(
    map(get_participant_number, PARTICIPANT_DIRNAMES_WITH_EXCEL)
)

# %%
validation_participants = random_state.choice(
    a=PARTICPANT_NUMBERS_WITH_EXCEL, size=validation_size, replace=False
)
validation_participants = set(validation_participants)

# %%
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
# %%
train_data = data.filter(items=train_columns).T
val_data = data.filter(items=val_columns).T

# %%
timeseries_length = len(data)
autoencoder = MLPDenoisingAutoEncoder(timeseries_length=timeseries_length)
autoencoder.compile(optimizer="adam", loss="mae")

# %%
autoencoder.build(train_data.shape)
autoencoder.summary()


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
model_directory = "models"


class StoreModelHistory(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        if not ("model_history.csv" in os.listdir(model_directory)):
            with open(os.path.join(model_directory, "model_history.csv"), "a") as f:
                y = csv.DictWriter(f, logs.keys())
                y.writeheader()

        with open(model_directory + "model_history.csv", "a") as f:
            y = csv.DictWriter(f, logs.keys())
            y.writerow(logs)


# %%

num_epochs = 10000
batch_size = 32

# Start a run, tracking hyperparameters
wandb.init(
    project="keras-intro",
    # Set entity to specify your username or team name
    # ex: entity="carey",
    config={
        "layer_1": 512,
        "activation_1": "relu",
        "layer_2": 10,
        "activation_2": "softmax",
        "optimizer": "sgd",
        "loss": "sparse_categorical_crossentropy",
        "metric": "accuracy",
        "epoch": 6,
        "batch_size": 32,
    },
)
config = wandb.config

history = autoencoder.fit(
    train_data,
    train_data,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(val_data, val_data),
    # callbacks=[early_stop],
    shuffle=True,
)

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
