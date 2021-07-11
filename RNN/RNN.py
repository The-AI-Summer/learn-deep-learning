from tensorflow.contrib.keras.python.keras.datasets import imdb
from tensorflow.contrib.keras.python.keras.layers import Embedding, SimpleRNN, Dropout, Dense, Activation, LSTM, GRU
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.preprocessing import sequence

max_features = 20000
maxlen = 100
batch_size = 32


(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))

#model.add(SimpleRNN(128))
#model.add(GRU(128))
model.add(LSTM(128))

model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam')


model.fit(X_train, y_train, batch_size=batch_size, epochs=1,
          validation_data=(X_test, y_test))