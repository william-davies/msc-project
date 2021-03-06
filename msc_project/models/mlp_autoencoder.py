from typing import Dict
import tensorflow as tf
from tensorflow import keras


def create_autoencoder(config: Dict):
    """

    :param config:
    :return:
    """
    bottleneck_size = 4
    autoencoder = tf.keras.Sequential(
        [
            # encoder
            keras.layers.Dense(
                units=bottleneck_size,
                activation="relu",
                input_shape=(config["timeseries_length"],),
            ),
            # decoder
            keras.layers.Dense(
                units=config["timeseries_length"],
                activation="sigmoid",
                input_shape=(bottleneck_size,),
            ),
        ]
    )

    autoencoder.compile(
        optimizer=config["optimizer"], loss=config["loss"], metrics=[config["metric"]]
    )

    return autoencoder


def reshape_data(data):
    """
    MLP is fine with the original (samples, timesteps) shape. A bit ugly that I have this pointless function but=
    does the job for now.
    :param data:
    :return:
    """
    return data
