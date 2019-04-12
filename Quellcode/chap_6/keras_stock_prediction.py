#
# Vorhersage von einem Aktienkurs mit Keras und LSTMs
#

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Input, LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
import os
import math
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


#https://www.macrotrends.net/stocks/charts/TSLA/tesla/stock-price-history
# 'date' 'open' 'high' 'low' 'close' 'volume'
#t = np.loadtxt("tsla.csv",delimiter=",",skiprows=9,comments="#",usecols=(0,4))
#print(t)

initial_stock_data = []

# Datei, die die 
CSV_FILE  = "data/aapl.csv" # "tsla.csv" 

tsla_stock_values = np.loadtxt(CSV_FILE,delimiter=",",skiprows=9,usecols=(4),comments="#",dtype=float)

date_values = np.loadtxt(CSV_FILE,delimiter=",",skiprows=9,usecols=(0),comments="#",dtype=[('date', "datetime64[D]")])
initial_stock_data = np.array(tsla_stock_values,dtype="float").reshape(-1,1) # Wir nehmen nur die Spalte (4)

# Normalisierung der Werte
min_max_scaler = MinMaxScaler(feature_range=(0,1))
stock_data = min_max_scaler.fit_transform(initial_stock_data)

# Speichert die historische Daten 
days_before_values = [] # T- days
day_values = [] # T 

DAYS_BEFORE = 20 # Anzahl der Tage in der Vergangenheit

# Reorganisiert die Daten
def arrange_data(stock_data, days):
    for i in range(len(stock_data) - days -1):
        days_before_values.append(stock_data[i:(i+days)]) 
        day_values.append(stock_data[i + days + 1]) # T 

# Wir generieren die Arrays days_before_values und days_values 
arrange_data(stock_data,DAYS_BEFORE)

# Splitting für die Evaluation 80% Training, 20% Test
# Beachten Sie shuffle=False, ansonsten ist die Zeitserie nicht mehr korrekt 
days_before_values_train, days_before_values_test, next_day_values_train, next_day_values_test = train_test_split(days_before_values,
                                                            day_values, 
                                                            test_size=0.20,
                                                            shuffle = False)

# Wir bauen ein Testdataset auf
start_test_index = len(days_before_values_train)
end_test_index = len(days_before_values_train) + 20 # Die nächsten 20 Tage

# Definition des Keras Modells
stock_model = Sequential()
stock_model.add(LSTM(24,input_shape=(DAYS_BEFORE,1),return_sequences=False))
stock_model.add(Dense(1))

sgd = SGD(lr=0.01)

stock_model.compile(loss="mean_squared_error", optimizer=sgd, metrics=[metrics.mse])

stock_model.fit(x=[days_before_values_train], y=[next_day_values_train], batch_size=10, epochs=100, verbose=1)

# Das Modell wird gespeichert
stock_model.save("keras_stock.h5")

# Evaluation der Testdaten
score, _ = stock_model.evaluate(x=[days_before_values_test], y=[next_day_values_test])

rmse = math.sqrt(score)

print("RMSE {}".format(rmse))

# Vorhersage basiert auf dem Trainingsdatenset
train_predict = stock_model.predict([days_before_values_train])
train_predict = min_max_scaler.inverse_transform(train_predict)
train_real = initial_stock_data[0:len(train_predict)].flatten()

# Train
train_score = math.sqrt(mean_squared_error(train_real,train_predict))
print("Train RMSE {}".format(train_score))

test_predict = stock_model.predict([days_before_values_test],verbose=1)
test_predict = min_max_scaler.inverse_transform(test_predict)

# Wir shiften nach rechts damit das Testergebnis grafisch direkt nach der Trainingskurve
# startet

rest = len(stock_data) - len(train_predict)
shift = range(len(train_predict)-1, len(stock_data) - 1 - DAYS_BEFORE - 1)

# Anzeige der Kurven mit matplotlib
plt.plot(initial_stock_data, color="#CFCEC4",label="Kurs")
plt.plot(train_predict, label="Training", color="green")
plt.plot(shift,test_predict,color="red", dashes=[6, 2], label="Test")
plt.legend(loc='upper left')
plt.show()