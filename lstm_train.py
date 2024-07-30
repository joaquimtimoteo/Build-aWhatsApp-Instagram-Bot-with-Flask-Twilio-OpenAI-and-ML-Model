import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Configurações
num_words = 10000  # Definir o número máximo de palavras
maxlen = 100  # Definir o comprimento máximo das sequências

# Carregar e preprocessar dados
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Definir o modelo LSTM
model_lstm = Sequential()
model_lstm.add(Embedding(input_dim=num_words, output_dim=128, input_length=maxlen))
model_lstm.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dense(1, activation='sigmoid'))

# Compilar o modelo
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model_lstm.fit(x_train, y_train, epochs=3, batch_size=64, validation_data=(x_test, y_test))

# Salvar o modelo treinado
model_lstm.save('models/lstm/model_lstm.h5')
