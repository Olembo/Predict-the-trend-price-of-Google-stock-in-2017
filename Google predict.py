#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 19:02:47 2020

@author: richmond
"""

# Reccurent Neural Network
#Partie 1: Preparation des donnees

#Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Training Dataset

dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train[["Open"]].values

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#Creation de la structure avc 60 timesteps et 1 sortie
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[(i-60):i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train = np.array(X_train)
y_train = np.array(y_train)

#Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Partie 2: Construction du RNN

#Librairies
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialisation
regressor = Sequential()

#couche LSTM + Dropout
regressor.add(LSTM(units= 50, return_sequences=True , input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# 2eme couche LSTM + Dropout
regressor.add(LSTM(units= 50, return_sequences=True))
regressor.add(Dropout(0.2))

# 3eme couche LSTM + Dropout
regressor.add(LSTM(units= 50, return_sequences=True))
regressor.add(Dropout(0.2))

# 4eme couche LSTM + Dropout
regressor.add(LSTM(units= 50))
regressor.add(Dropout(0.2))

#couche de sortir 
regressor.add(Dense(units=1))

# Compilation
regressor.compile(optimizer="adam", loss="mean_squared_error")

# Entrainement
regressor.fit(X_train, y_train, epochs=100, batch_size=32)


#Partie 3: Predictions et visualisation

#Donnees de 2017
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test[["Open"]].values

# Predictions pour 2017
dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis=0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[(i-60):i, 0])
X_test = np.array(X_test)


predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualisation des resultats
plt.plot(real_stock_price, color="red", label="Prix reel de l'action Google")
plt.plot(predicted_stock_price, color="green", label="Prix predit de l'action Google")
plt.title("Prediction de l'action Google")
plt.xlabel("Jour")
plt.ylabel("Prix de l'action")
plt.legend()
plt.show()