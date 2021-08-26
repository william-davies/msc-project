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
latent_encoding = MaxPool1D(2)(latent_encoding)
latent_encoding = Conv1D(4, 3, activation="relu", padding="same")(latent_encoding)
latent_encoding = MaxPool1D(2)(latent_encoding)
latent_encoding = Flatten()(latent_encoding)
latent_encoding = Dense(2)(latent_encoding)

# decoder
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
model = keras.Model(input_signal, decoded_output)

plot_model(model, show_shapes=True, to_file="example_cnn.png")
print(model.summary())

# %%
input_window = Input(shape=(timesteps, num_features))
x = Conv1D(16, 3, activation="relu", padding="same")(input_window)  # 10 dims
# x = BatchNormalization()(x)
x = MaxPool1D(2, padding="same")(x)  # 5 dims
x = Conv1D(1, 3, activation="relu", padding="same")(x)  # 5 dims
# x = BatchNormalization()(x)
encoded = MaxPool1D(2, padding="same")(x)  # 3 dims

# 3 dimensions in the encoded layer

x = Conv1D(1, 3, activation="relu", padding="same")(encoded)  # 3 dims
# x = BatchNormalization()(x)
x = UpSampling1D(2)(x)  # 6 dims
x = Conv1D(16, 2, activation="relu")(x)  # 5 dims
# x = BatchNormalization()(x)
x = UpSampling1D(2)(x)  # 10 dims
decoded = Conv1D(1, 3, activation="sigmoid", padding="same")(x)  # 10 dims
autoencoder = keras.Model(input_window, decoded)
plot_model(autoencoder, show_shapes=True, to_file="example_cnn.png")
autoencoder.summary()
