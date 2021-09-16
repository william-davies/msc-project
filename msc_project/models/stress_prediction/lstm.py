from typing import Dict
import tensorflow as tf
from keras import Sequential, Input
from keras.layers import LSTM, Dense
from tensorflow import keras


def instantiate_predictor(config: Dict):
    """

    :param config:
    :return:
    """
    num_features = 1
    autoencoder = Sequential()
    autoencoder.add(Input(shape=(config["timeseries_length"], num_features)))
    autoencoder.add(LSTM(16))
    autoencoder.add(Dense(1, activation="sigmoid"))
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
    lstm = instantiate_predictor(config=config)
    lstm.summary()
