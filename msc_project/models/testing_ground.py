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


input_sig = Input(batch_shape=(1, 128, 1))
x = Conv1D(8, 3, activation="relu", padding="same", dilation_rate=2)(input_sig)
x1 = MaxPool1D(2)(x)
x2 = Conv1D(4, 3, activation="relu", padding="same", dilation_rate=2)(x1)
x3 = MaxPool1D(2)(x2)
x4 = AveragePooling1D()(x3)
flat = Flatten()(x4)
encoded = Dense(2)(flat)
d1 = Dense(64)(encoded)
d2 = Reshape((16, 4))(d1)
d3 = Conv1D(4, 1, strides=1, activation="relu", padding="same")(d2)
d4 = UpSampling1D(2)(d3)
d5 = Conv1D(8, 1, strides=1, activation="relu", padding="same")(d4)
d6 = UpSampling1D(2)(d5)
d7 = UpSampling1D(2)(d6)
decoded = Conv1D(1, 1, strides=1, activation="sigmoid", padding="same")(d7)
model = keras.Model(input_sig, decoded)

plot_model(model, show_shapes=True, to_file="example_cnn.png")
print(model.summary())
