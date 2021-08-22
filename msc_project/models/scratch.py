# lstm autoencoder recreate sequence
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.utils import plot_model

# define input sequence
sequence = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# reshape input into [samples, timesteps, features]
n_in = len(sequence)
sequence = sequence.reshape((1, n_in, 1))
# define model
model = Sequential()
model.add(LSTM(100, activation="relu", input_shape=(n_in, 1)))
model.add(RepeatVector(n_in))
model.add(LSTM(100, activation="relu", return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer="adam", loss="mse")
# fit model
model.fit(sequence, sequence, epochs=300, verbose=0)
plot_model(model, show_shapes=True, to_file="reconstruct_lstm_autoencoder.png")
# demonstrate recreation
yhat = model.predict(sequence, verbose=0)
print(yhat[0, :, 0])
