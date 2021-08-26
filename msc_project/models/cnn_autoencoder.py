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
from lstm_autoencoder import reshape_data


def create_autoencoder(config: Dict):
    """

    :param config:
    :return:
    """
    num_features = 1

    input_signal = Input(shape=(config["timeseries_length"], num_features))
    x = Conv1D(16, 5, activation="relu", padding="same")(input_signal)
    x = MaxPool1D(4, padding="same")(x)
    x = Conv1D(1, 3, activation="relu", padding="same")(x)
    encoded = MaxPool1D(4, padding="same", name="final_encoder_layer")(x)

    x = Conv1D(1, 3, activation="relu", padding="same")(encoded)
    x = UpSampling1D(4)(x)
    x = Conv1D(16, 2, activation="relu", padding="same")(x)
    x = UpSampling1D(4)(x)
    decoded = Conv1D(1, 3, activation="sigmoid", padding="same")(x)
    autoencoder = keras.Model(input_signal, decoded)

    autoencoder.compile(
        optimizer=config["optimizer"], loss=config["loss"], metrics=[config["metric"]]
    )

    return autoencoder
