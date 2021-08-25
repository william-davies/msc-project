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
)

num_features = 1
latent_dimension = 4
timesteps = 128

# encoder
encoder_input = keras.Input(shape=(timesteps, num_features))
latent_encoding = MaxPool1D(
    pool_size=16, strides=16, padding="valid", data_format="channels_last"
)(encoder_input)
# latent_encoding = Conv1D(filters=5, kernel_size=45, activation='relu')(latent_encoding)
AveragePooling1D()

autoencoder = keras.Model(inputs=encoder_input, outputs=latent_encoding)
autoencoder.compile(optimizer="adam", loss="mae")
plot_model(autoencoder, show_shapes=True, to_file="cnn_autoencoder.png")
