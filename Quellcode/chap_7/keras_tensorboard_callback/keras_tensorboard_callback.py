# 
# Anwendung eines TensorBoard Callbacks mit Keras
#


import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data 
from keras import *
from keras.layers import *
from keras.utils import np_utils
from keras.initializers import Constant
from keras.callbacks import RemoteMonitor, LambdaCallback, TensorBoard
import requests as requests
from keras import metrics

# Fashion Klassen
fashion_class_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

model = Sequential()

# Labels und Daten werden hier geladen
def load_fashion_data():
    data = input_data.read_data_sets('data/fashion') #,one_hot=True)

    # Trainingsdaten
    train_data = data.train.images
    train_labels = data.train.labels
    train_labels = np.asarray(data.train.labels,dtype=np.int32)
    
    # Evaluationsdaten
    eval_data = data.test.images  
    eval_labels = data.test.labels
    eval_labels = np.asarray(data.test.labels,dtype=np.int32)
    eval_data, eval_labels = shuffle(eval_data, eval_labels)
    return (train_data, train_labels, eval_data, eval_labels) 

# Laden der Daten
train_data, train_labels, eval_data, eval_labels = load_fashion_data()
train_data = train_data.reshape(-1, 28, 28, 1)
train_labels = np_utils.to_categorical(train_labels, 10)

# Model mit Keras
model.add(InputLayer(input_shape=(28, 28,1),name="1_Eingabe"))
model.add(BatchNormalization(name="2_Batch_Normalisierung"))
model.add(Conv2D(32,(2, 2),padding='same',bias_initializer=Constant(0.01),kernel_initializer='random_uniform',name="3_Conv2D"))
model.add(Activation(activation='relu',name="4_ReLu"))
model.add(MaxPool2D(padding='same',name="5_MaxPooling2D"))
model.add(Conv2D(32,(2, 2),padding='same',bias_initializer=Constant(0.01),kernel_initializer='random_uniform',name="6_Conv2D"))
model.add(Activation(activation="relu",name='7_ReLu'))
model.add(MaxPool2D(padding='same',name="8_MaxPooling2D"))
model.add(Flatten())
model.add(Dense(1024,activation='relu',bias_initializer=Constant(0.01),kernel_initializer='random_uniform',name="9_Dense"))
model.add(Dropout(0.4,name="10_Dense"))
model.add(Dense(10, activation='softmax',name="11_Ausgabe"))
model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adadelta(), metrics = ["accuracy","mse",metrics.categorical_accuracy])

# TensorBoard Callback
tf_cb = callbacks.TensorBoard(log_dir='./logs', batch_size=64, write_graph=True)
model.fit(train_data,train_labels, batch_size=64, epochs=100, verbose=1,validation_split=0.33,callbacks=[tf_cb])
