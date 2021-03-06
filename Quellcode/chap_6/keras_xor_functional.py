#
# XOR-Modell Implementierung mit Keras Functional API
#

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import SGD

from pprint import pprint
import numpy as np

# Einfaches XOR mit Keras
input_data = np.array([[0,0],[0,1],[1,0],[1,1]])
output_data = np.array([[0],[1],[1],[0]])

inputs = Input(shape=(2,),dtype="float32")
x = Dense(1024,name="First_Layer")(inputs)
x = Activation('relu',name="Relu_Activation")(x)
x = Dense(1,name="Dense_Layer")(x)
predictions = Activation('sigmoid',name="Sigmoid_Activation")(x)

sgd = SGD(lr=0.01)
xor_model = Model(inputs=inputs,outputs=predictions)
xor_model.compile(loss="mean_squared_error", optimizer=sgd)
xor_model.fit(input_data, output_data, batch_size=1, epochs=3000, verbose=1)
print(xor_model.predict(input_data))

