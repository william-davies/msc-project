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

input_signal = keras.Input(shape=(timesteps, num_features))

# encoder
latent_encoding = Conv1D(8, 3, activation="relu", padding="same")(input_signal)
latent_encoding = MaxPool1D(2, padding="same")(latent_encoding)
latent_encoding = Conv1D(4, 3, activation="relu", padding="same")(latent_encoding)
latent_encoding = MaxPool1D(2, padding="same")(latent_encoding)

# decoder
decoded_output = Conv1D(4, 1, strides=1, activation="relu", padding="same")(
    latent_encoding
)
decoded_output = UpSampling1D(2)(decoded_output)
decoded_output = Conv1D(8, 1, strides=1, activation="relu", padding="same")(
    decoded_output
)
decoded_output = UpSampling1D(4)(decoded_output)
decoded_output = Conv1D(1, 1, strides=1, activation="sigmoid", padding="same")(
    decoded_output
)
model = keras.Model(input_signal, decoded_output)

plot_model(model, show_shapes=True, to_file="example_cnn.png")
print(model.summary())

# %%
timesteps = 128

input_signal = Input(shape=(timesteps, num_features))
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
plot_model(autoencoder, show_shapes=True, to_file="example_cnn.png")
autoencoder.summary()
