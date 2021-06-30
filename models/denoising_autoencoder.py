from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers


class MLPDenoisingAutoEncoder(Model):
    def __init__(self, timeseries_length):
        super(MLPDenoisingAutoEncoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Dense(8192, activation="relu", input_shape=(timeseries_length,)),
                layers.Dense(4096, activation="relu"),
                layers.Dense(2048, activation="relu"),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(4096, activation="relu"),
                layers.Dense(8192, activation="relu"),
                layers.Dense(timeseries_length, activation="sigmoid"),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
