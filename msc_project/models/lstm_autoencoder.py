from typing import Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import (
    RepeatVector,
    Bidirectional,
    Dense,
    LSTM,
    TimeDistributed,
)


# def create_autoencoder(config: Dict):
#     """
#
#     :param config:
#     :return:
#     """
#     bottleneck_size = 8
#     autoencoder = tf.keras.Sequential(
#         [
#             # encoder
#             keras.layers.LSTM(
#                 units=bottleneck_size,
#                 activation="tanh",
#                 input_shape=(config["timeseries_length"], 1),
#             ),
#             RepeatVector(config["timeseries_length"]),
#             # decoder
#             keras.layers.LSTM(
#                 units=bottleneck_size, activation="tanh", return_sequences=True
#             ),
#             keras.layers.TimeDistributed(keras.layers.Dense(1, activation="sigmoid")),
#         ]
#     )
#
#     autoencoder.compile(
#         optimizer=config["optimizer"], loss=config["loss"], metrics=[config["metric"]]
#     )
#
#     return autoencoder


def create_autoencoder(config: Dict):
    """

    :param config:
    :return:
    """
    num_features = 1
    latent_dimension = 4

    # encoder
    encoder_input = keras.Input(shape=(config["timeseries_length"], num_features))
    latent_encoding = Bidirectional(LSTM(latent_dimension, activation="tanh"))(
        encoder_input
    )

    latent_encoding = RepeatVector(config["timeseries_length"])(latent_encoding)

    # decoder
    decoder_output = LSTM(
        latent_dimension * 2, activation="tanh", return_sequences=True
    )(latent_encoding)
    decoder_output = TimeDistributed(Dense(units=1, activation="sigmoid"))(
        decoder_output
    )

    autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
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


if __name__ == "__main__":
    # just for development
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
    plot_model(autoencoder, show_shapes=True, to_file="blstm_autoencoder.png")
    print(autoencoder.summary())
