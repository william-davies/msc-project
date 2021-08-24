# lstm autoencoder recreate sequence
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.utils import plot_model
from tensorflow import keras
from tensorflow.keras import layers

# define input sequence
from msc_project.models.lstm_autoencoder import create_autoencoder

sequence = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# reshape input into [samples, timesteps, features]
timesteps = len(sequence)
sequence = sequence.reshape((1, timesteps, 1))

num_features = 1
latent_dimension = 4

api_type = "functional"

if api_type == "sequential":
    autoencoder = Sequential()
    autoencoder.add(
        Bidirectional(
            LSTM(
                latent_dimension,
                activation="tanh",
                input_shape=(timesteps, num_features),
            )
        )
    )
    autoencoder.add(RepeatVector(timesteps))
    autoencoder.add(
        LSTM(latent_dimension * 2, activation="tanh", return_sequences=True)
    )
    autoencoder.add(TimeDistributed(Dense(1, activation="sigmoid")))

elif api_type == "functional":
    # encoder_inputs = keras.Input(shape=(timesteps, num_features))
    # encoder_blstm = keras.layers.Bidirectional(keras.layers.LSTM(latent_dimension, return_state=False))
    # latent_encoding = encoder_blstm(
    #     encoder_inputs
    # )
    # repeat_vector = RepeatVector(timesteps)(latent_encoding)
    # # decoder
    # decoder_lstm = keras.layers.LSTM(latent_dimension * 2, return_sequences=True)
    # decoder_outputs = decoder_lstm(repeat_vector)
    # decoder_dense = keras.layers.TimeDistributed(keras.layers.Dense(units=1, activation='softmax'))
    # decoder_outputs = decoder_dense(decoder_outputs)
    #
    # autoencoder = keras.Model(inputs=encoder_inputs, outputs=decoder_outputs)

    encoder_input = keras.Input(shape=(timesteps, num_features))
    x = Bidirectional(LSTM(latent_dimension, activation="tanh"))(encoder_input)

    x = RepeatVector(timesteps)(x)

    x = LSTM(latent_dimension * 2, activation="tanh", return_sequences=True)(x)
    decoder_output = TimeDistributed(Dense(units=1, activation="sigmoid"))(x)

    autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")

autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(sequence, sequence, epochs=300, verbose=1)
print(autoencoder.summary())
plot_model(autoencoder, show_shapes=True, to_file=f"{api_type}.png")
# demonstrate recreation
yhat = autoencoder.predict(sequence, verbose=0)
print(yhat[0, :, 0])
