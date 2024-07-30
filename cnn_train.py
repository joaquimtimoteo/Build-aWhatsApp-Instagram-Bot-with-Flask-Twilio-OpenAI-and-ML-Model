import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

# Configurações
num_words = 10000
maxlen = 100

# Carregar e preprocessar dados
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Definir o modelo CNN
model_cnn = Sequential()
model_cnn.add(Embedding(input_dim=num_words, output_dim=128, input_length=maxlen))
model_cnn.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model_cnn.add(GlobalMaxPooling1D())
model_cnn.add(Dense(1, activation='sigmoid'))

# Compilar o modelo
model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model_cnn.fit(x_train, y_train, epochs=3, batch_size=64, validation_data=(x_test, y_test))

# Salvar o modelo
model_cnn.save('models/model_cnn.h5')
