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
    Flatten,
    UpSampling1D,
    Reshape,
)
from keras import Input

num_features = 1
latent_dimension = 4
timesteps = 128

# encoder
encoder_input = keras.Input(shape=(timesteps, num_features))
latent_encoding = Conv1D(filters=4, kernel_size=3, activation="relu")(encoder_input)
latent_encoding = MaxPool1D(pool_size=3, padding="valid", data_format="channels_last")(
    latent_encoding
)
# latent_encoding = AveragePooling1D(pool_size=2)(latent_encoding)
latent_encoding = Flatten()(latent_encoding)

# decoder_output = Conv1D(filters=5, kernel_size=1, strides=1)(latent_encoding)
# decoder_output = UpSampling1D(2)(decoder_output)


autoencoder = keras.Model(inputs=encoder_input, outputs=latent_encoding)
autoencoder.compile(optimizer="adam", loss="mae")
plot_model(autoencoder, show_shapes=True, to_file="cnn_autoencoder.png")
print(autoencoder.summary())


encoder_input = keras.Input(shape=(timesteps, num_features))
latent_encoding = Conv1D(8, 3, activation="relu", padding="same", dilation_rate=2)(
    encoder_input
)
latent_encoding = MaxPool1D(2)(latent_encoding)
latent_encoding = Conv1D(4, 3, activation="relu", padding="same", dilation_rate=2)(
    latent_encoding
)
latent_encoding = MaxPool1D(2)(latent_encoding)
latent_encoding = Flatten()(latent_encoding)
latent_encoding = Dense(2)(latent_encoding)

decoded_output = Dense(64)(latent_encoding)
decoded_output = Reshape((16, 4))(decoded_output)
decoded_output = Conv1D(4, 1, strides=1, activation="relu", padding="same")(
    decoded_output
)
decoded_output = UpSampling1D(2)(decoded_output)
decoded_output = Conv1D(8, 1, strides=1, activation="relu", padding="same")(
    decoded_output
)
decoded_output = UpSampling1D(4)(decoded_output)
decoded_output = Conv1D(1, 1, strides=1, activation="sigmoid", padding="same")(
    decoded_output
)
model = keras.Model(encoder_input, decoded_output)

plot_model(model, show_shapes=True, to_file="example_cnn.png")
print(model.summary())

# %%
import numpy as np
import tensorflow as tf

input_shape = (2, 2, 3)
x = np.arange(np.prod(input_shape)).reshape(input_shape)
print(x)

y1 = tf.keras.layers.UpSampling1D(size=2)(x)
y1 = tf.keras.layers.UpSampling1D(size=2)(y1)
print(y1)

y2 = tf.keras.layers.UpSampling1D(size=4)(x)
print(y2)
