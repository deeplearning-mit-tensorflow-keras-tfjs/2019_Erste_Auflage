#
# Erstellung eines Bildklassifikators auf Basis des CIFAR-10 Dataset mit Keras 
#

import tensorflow as tf
import numpy as np
import urllib
import matplotlib
import os
import math
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from tensorflow.python import keras
from tensorflow.python.keras.datasets import cifar10

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Activation, InputLayer, LSTM, BatchNormalization, Conv2D, MaxPool2D, Flatten
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from PIL import Image

# Variablen für das Training
BATCH_SIZE = 64#32
EPOCHS = 20 #50

# Für die 10 Klassen von CIFAR-10
CIFAR_10_CLASSES = ["Flugzeug","Fahrzeug","Vogel","Katze","Wild","Hund", "Frosch", "Pferd","Boot","LKW"]
NUM_CLASSES = 10

# Wir laden den Datenset über Keras
(images_train, labels_train), (images_test, labels_test) = cifar10.load_data()

# Test
plt.title(CIFAR_10_CLASSES[int(labels_train[25])])
plt.imshow(images_train[25])
plt.show()

images_train = np.array(images_train,dtype="float32")
images_test = np.array(images_test,dtype="float32")

images_train /= 255 # Damit die Werte zwischen 0 und 1 bleiben
images_test /=255

labels_train = to_categorical(labels_train, NUM_CLASSES)
labels_test = to_categorical(labels_test, NUM_CLASSES)

# Definition des Modells 
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=(32, 32, 3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.add(Dense(NUM_CLASSES,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics = ["accuracy"])
model.fit(images_train,labels_train, batch_size=BATCH_SIZE,epochs=EPOCHS)

scores = model.evaluate(images_test,labels_test)

print('Loss:', scores[0])
print('Accuracy:', scores[1])

# Wir speichern das Model. In keras_cnn_test_cifar.py wird dieses geladen
model.save("cifar_model.h5")