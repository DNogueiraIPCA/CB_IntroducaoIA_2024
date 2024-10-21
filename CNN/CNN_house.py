# -*- coding: utf-8 -*-
"""
Created on Sat May 27 06:03:41 2023

@author: Daniel Nogueira
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Carregamento do conjunto de dados Boston Housing
(x_treino, y_treino), (x_teste, y_teste) = boston_housing.load_data()

# Normalização dos dados
mean = x_treino.mean(axis=0)
std = x_treino.std(axis=0)
x_treino = (x_treino - mean) / std
x_teste = (x_teste - mean) / std

# Criação do modelo da CNN
modelo = Sequential()
modelo.add(Conv1D(32, 3, activation='relu', input_shape=(x_treino.shape[1], 1)))
modelo.add(MaxPooling1D(2))
modelo.add(Flatten())
modelo.add(Dense(64, activation='relu'))
modelo.add(Dense(1))

# Compilação do modelo
modelo.compile(optimizer='adam', loss='mean_squared_error')

# Treinamento do modelo
modelo.fit(x_treino, y_treino, epochs=50, batch_size=32, validation_data=(x_teste, y_teste))

# Avaliação do modelo
loss = modelo.evaluate(x_teste, y_teste)
print('Loss:', loss)

predict = modelo.predict(x_teste)
pred    = list()
for i in range(len(predict)):
    pred.append(predict[i][0])

df_pred = pd.DataFrame(pred, columns=['Pred'])
df_pred['Real'] = y_teste

plt.plot(df_pred.Pred)
plt.plot(df_pred.Real)
plt.title('Valor Real x Valor Predito')
plt.ylabel('% Aprovação')
plt.xlabel('Dados')
plt.legend(['Predito', 'Real'])
plt.show()
