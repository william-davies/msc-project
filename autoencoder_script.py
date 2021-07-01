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

print(validation_participants)

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
normalised_train_data = (train_data.values - min_val) / (max_val - min_val)
normalised_val_data = (val_data.values - min_val) / (max_val - min_val)

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

history = autoencoder.fit(
    train_data.T,
    train_data.T,
    epochs=num_epochs,
    batch_size=16,
    validation_data=(val_data.T, val_data.T),
    # callbacks=[early_stop],
    shuffle=True,
)

# %%
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

# %%
example = train_data.T.iloc[0].values

# %%
decoded_examples = tf.stop_gradient(autoencoder.call(train_data.T.values))

# %%
plt.plot(train_data.T.iloc[0], "b")
plt.plot(decoded_examples[0], "r")
plt.show()
