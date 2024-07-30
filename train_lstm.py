import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical

# Parâmetros
max_features = 10000  # Número máximo de palavras a serem consideradas
maxlen = 100  # Número máximo de palavras em cada sequência
embedding_dim = 128  # Dimensão da camada de embedding
lstm_units = 128  # Número de unidades na camada LSTM

# Carregar e preparar os dados
print("Carregando dados...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Padronizar as sequências para que todas tenham o mesmo comprimento
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Convertendo rótulos para uma forma categórica (one-hot encoding)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Criar o modelo LSTM
print("Construindo o modelo...")
model = Sequential()
model.add(Embedding(max_features, embedding_dim, input_length=maxlen))
model.add(LSTM(lstm_units))
model.add(Dense(2, activation='softmax'))

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
print("Treinando o modelo...")
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Avaliar o modelo
print("Avaliando o modelo...")
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Perda: {loss:.4f}")
print(f"Acurácia: {accuracy:.4f}")

# Salvar o modelo
print("Salvando o modelo...")
model.save('models/model_lstm.h5')

print("Modelo LSTM treinado e salvo com sucesso.")
