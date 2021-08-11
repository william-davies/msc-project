from typing import Dict

from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers


def create_autoencoder(config: Dict):
    """

    :param config:
    :return:
    """
    autoencoder = tf.keras.Sequential(
        [
            # encoder
            layers.Dense(
                config["encoder_1"],
                activation=config["encoder_activation_1"],
                input_shape=(config["timeseries_length"],),
            ),
            layers.Dense(
                config["encoder_2"], activation=config["encoder_activation_2"]
            ),
            layers.Dense(
                config["encoder_3"], activation=config["encoder_activation_3"]
            ),
            # decoder
            layers.Dense(
                config["decoder_1"],
                activation=config["decoder_activation_1"],
                input_shape=(config["encoder_3"],),
            ),
            layers.Dense(
                config["decoder_2"], activation=config["decoder_activation_2"]
            ),
            layers.Dense(
                config["decoder_3"], activation=config["decoder_activation_3"]
            ),
        ]
    )

    autoencoder.compile(
        optimizer=config["optimizer"], loss=config["loss"], metrics=[config["metric"]]
    )

    return autoencoder
