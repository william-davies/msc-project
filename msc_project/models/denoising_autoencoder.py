import re
from typing import Dict

from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow import keras


def create_autoencoder(config: Dict):
    """

    :param config:
    :return:
    """

    def get_layers(pattern, units_key_placeholder, activation_key_placeholder):
        layers = []
        for key in config.keys():
            if pattern.match(key):
                layer_number = int(pattern.match(key).group(1))
                if layer_number > 1:
                    layer = keras.layers.Dense(
                        config[units_key_placeholder.format(layer_number)],
                        activation=config[
                            activation_key_placeholder.format(layer_number)
                        ],
                    )
                    layers.append(layer)
        return layers

    encoder_layers = [
        keras.layers.Dense(
            config["encoder_1"],
            activation=config["encoder_activation_1"],
            input_shape=(config["timeseries_length"],),
        )
    ]
    encoder_layers.extend(
        get_layers(
            pattern=re.compile("encoder_(\d)"),
            units_key_placeholder="encoder_{}",
            activation_key_placeholder="encoder_activation_{}",
        )
    )

    decoder_layers = [
        keras.layers.Dense(
            config["decoder_1"],
            activation=config["decoder_activation_1"],
            input_shape=(encoder_layers[-1].units,),
        )
    ]
    decoder_layers.extend(
        get_layers(
            pattern=re.compile("decoder_(\d)"),
            units_key_placeholder="decoder_{}",
            activation_key_placeholder="decoder_activation_{}",
        )
    )

    autoencoder = tf.keras.Sequential(
        [
            # encoder
            *encoder_layers,
            # decoder
            *decoder_layers,
        ]
    )

    autoencoder.compile(
        optimizer=config["optimizer"], loss=config["loss"], metrics=[config["metric"]]
    )

    return autoencoder
