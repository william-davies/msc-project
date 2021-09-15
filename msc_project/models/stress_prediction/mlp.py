from typing import Dict
import tensorflow as tf
from tensorflow import keras


def instantiate_predictor(config: Dict):
    """

    :param config:
    :return:
    """
    bottleneck_size = 8
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
                units=1,
                activation="sigmoid",
                input_shape=(bottleneck_size,),
            ),
        ]
    )

    autoencoder.compile(
        optimizer=config["optimizer"], loss=config["loss"], metrics=[config["metric"]]
    )

    return autoencoder


if __name__ == "__main__":
    config = {
        "timeseries_length": 128,
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "metric": [None],
    }
    mlp = instantiate_predictor(config=config)
    mlp.summary()
