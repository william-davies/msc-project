import datetime
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from constants import PARTICIPANT_DIRNAMES_WITH_EXCEL, PARTICIPANT_NUMBER_PATTERN
from models.denoising_autoencoder import MLPDenoisingAutoEncoder

# %%

random_state = np.random.RandomState(42)

data = pd.read_csv("Stress Dataset/dataset_two_min_window.csv")

NUM_PARTICIPANTS = len(PARTICIPANT_DIRNAMES_WITH_EXCEL)

validation_size = round(NUM_PARTICIPANTS * 0.3)


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
train_columns = []
val_columns = []
for participant_column in data.columns:
    number_pattern = re.compile("^P(\d{1,2})_")
    participant_number = number_pattern.match(participant_column).group(1)
    if participant_number in validation_participants:
        val_columns.append(participant_column)
    else:
        train_columns.append(participant_column)
# %%
train_data = data.filter(items=train_columns)
val_data = data.filter(items=val_columns)

# %%
# Normalize the data to [0, 1]

min_val = tf.reduce_min(data)
max_val = tf.reduce_max(data)

# %%
normalised_data = (data - float(min_val)) / (float(max_val) - float(min_val))
normalised_data = normalised_data.values.T

# %%
normalised_train_data = (train_data.values - min_val) / (max_val - min_val)
normalised_train_data = tf.transpose(normalised_train_data)
normalised_train_data = normalised_train_data.numpy()

normalised_val_data = (val_data.values - min_val) / (max_val - min_val)
normalised_val_data = tf.transpose(normalised_val_data)
normalised_val_data = normalised_val_data.numpy()

# %%
plt.plot(tf.transpose(normalised_train_data)[10])
plt.show()

# %%
plt.plot(train_data.iloc[:, 10])
plt.show()
# %%
timeseries_length = len(data)
autoencoder = MLPDenoisingAutoEncoder(timeseries_length=timeseries_length)
autoencoder.compile(optimizer="adam", loss="mae")

# %%
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=1e-2,
    patience=5,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)

# %%
# history = autoencoder.fit(train_data.T, train_data.T,
#           epochs=1,
#           batch_size=16,
#           validation_data=(val_data.T, val_data.T),
#           callbacks=[early_stop],
#           shuffle=True)
num_epochs = 20

# history = autoencoder.fit(
#     train_data.T,
#     train_data.T,
#     epochs=num_epochs,
#     batch_size=16,
#     validation_data=(val_data.T, val_data.T),
#     # callbacks=[early_stop],
#     shuffle=True,
# )

history = autoencoder.fit(
    normalised_train_data,
    normalised_train_data,
    epochs=num_epochs,
    batch_size=16,
    validation_data=(normalised_val_data, normalised_val_data),
    # callbacks=[early_stop],
    shuffle=True,
)

# %%

model_save_fp = os.path.join(
    "models", "checkpoints", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
# model_save_fp = os.path.join('models', 'checkpoints')
autoencoder.save_weights(model_save_fp)

# %%
loaded_autoencoder = MLPDenoisingAutoEncoder(timeseries_length=timeseries_length)
loaded_autoencoder.compile(optimizer="adam", loss="mae")
loaded_autoencoder.load_weights(model_save_fp)
# %%
plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

# %%
example = train_data.T.iloc[0].values

# %%
decoded_train_examples = tf.stop_gradient(autoencoder.call(normalised_train_data))

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
decoded_all_examples = tf.stop_gradient(autoencoder.call(normalised_data))

# %%
plt.figure(figsize=(120, 20))
# plt.figure(figsize=(8, 6))

plt.plot(normalised_data[80], "b")
plt.plot(decoded_all_examples[80], "r")
plt.show()

# %%
plt.figure(figsize=(8, 6))
save_filepath = "autoencoder_plots/{}.png"

# for idx in range(len(normalised_data)):
for idx in range(10):
    title = data.columns[idx]
    plt.title(title)

    plt.plot(normalised_data[idx], "b")
    # plt.plot(decoded_all_examples[idx], "r")

    plt.xlabel("Frames")
    plt.ylabel("BVP")

    plt.savefig(save_filepath.format(title), format="png")
    plt.clf()
