from typing import Dict
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import (
    RepeatVector,
    Bidirectional,
    Dense,
    LSTM,
    TimeDistributed,
    MaxPool1D,
    Conv1D,
    AveragePooling1D,
    UpSampling1D,
)
from keras import Input


def create_autoencoder(config: Dict):
    """

    :param config:
    :return:
    """
    timesteps = config["timeseries_length"]
    num_features = 1

    input_signal = Input(shape=(timesteps, num_features))
    x = Conv1D(8, 16, activation="relu", padding="same")(input_signal)
    x = MaxPool1D(4, padding="same")(x)
    x = Conv1D(4, 4, activation="relu", padding="same")(x)
    x = MaxPool1D(4, padding="same")(x)

    x = Conv1D(4, 4, activation="relu", padding="same", name="decoder-1")(x)
    x = UpSampling1D(4)(x)
    x = Conv1D(8, 4, activation="relu", padding="same")(x)
    x = UpSampling1D(4)(x)
    decoded = Conv1D(1, 16, activation="sigmoid", padding="same")(x)
    autoencoder = keras.Model(input_signal, decoded)

    autoencoder.compile(
        optimizer=config["optimizer"], loss=config["loss"], metrics=[config["metric"]]
    )

    return autoencoder
