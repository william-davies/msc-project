from typing import Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import RepeatVector


def create_autoencoder(config: Dict):
    """

    :param config:
    :return:
    """
    bottleneck_size = 8
    autoencoder = tf.keras.Sequential(
        [
            # encoder
            keras.layers.LSTM(
                units=bottleneck_size,
                activation="tanh",
                input_shape=(config["timeseries_length"], 1),
            ),
            RepeatVector(config["timeseries_length"]),
            # decoder
            keras.layers.LSTM(
                units=bottleneck_size, activation="tanh", return_sequences=True
            ),
            keras.layers.TimeDistributed(keras.layers.Dense(1, activation="sigmoid")),
        ]
    )

    autoencoder.compile(
        optimizer=config["optimizer"], loss=config["loss"], metrics=[config["metric"]]
    )

    return autoencoder


def reshape_data(data):
    """
    LSTM likes shape (samples, timesteps, features)
    :param data:
    :return:
    """
    return data.values.reshape((*data.shape, 1))


run_config = {
    "optimizer": "adam",
    "loss": "mae",
    "metric": [None],
    "batch_size": 32,
    "monitor": "val_loss",
    "epoch": 10,
    "patience": 1000,
    "min_delta": 1e-3,
}

timeseries_length = 128
metadata = {
    **run_config,
    "timeseries_length": timeseries_length,
}
autoencoder = create_autoencoder(metadata)
plot_model(autoencoder, show_shapes=True, to_file="msc_autoencoder.png")
print(autoencoder.summary())
