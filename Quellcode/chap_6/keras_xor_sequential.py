#
# XOR-Modell mit Keras Sequential API
#
import tensorflow as tf
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import SGD
from pprint import pprint

# Einfaches XOR mit Keras
input_data = np.array([[0,0],[0,1],[1,0],[1,1]])
output_data = np.array([[0],[1],[1],[0]])

# Variante 1 mit Benennung der Schichten
''' 
xor_model = Sequential()
xor_model.add(Dense(1024, input_dim=2,name="First_Layer"))
xor_model.add(Activation('relu',name="Relu_Activation"))
xor_model.add(Dense(1,name="Dense_Layer"))
xor_model.add(Activation('sigmoid',name="Sigmoid_Activation"))
'''

# Variante 2
xor_model = Sequential()
xor_model.add(Dense(1024,input_dim=2,activation="relu"))
xor_model.add(Dense(1,activation="sigmoid"))


# Variante 3 als Array 
'''
xor_model = Sequential([
    Dense(1024, input_dim=2),
    Activation('relu'),
    Dense(1),
    Activation('sigmoid')
])
'''

xor_model.summary()
sgd = SGD(lr=0.01)

# Modell wird festgelegt und trainiert
xor_model.compile(loss="mean_squared_error", optimizer=sgd, metrics=["mae"])
xor_model.fit(input_data, output_data, batch_size=1, epochs=3000, verbose=1)
pprint(xor_model.predict(input_data))

