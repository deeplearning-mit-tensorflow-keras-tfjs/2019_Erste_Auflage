#
# Benutzung von Keras Callbacks zur Visualisierung des Modells und der Metriken
# Das Beispiel benutzt die Fashion-MNIST Klassifikationsaufgabe als Grundlage 
#

import tensorflow as tf
import numpy as np
import requests as requests

from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data 
from tensorflow.python import keras
from tensorflow.python.keras.layers import InputLayer, BatchNormalization, MaxPool2D, Conv2D,Flatten,Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import losses
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import metrics
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.callbacks import LambdaCallback, RemoteMonitor

# Labels und Daten werden hier geladen
def load_data():
    data = input_data.read_data_sets('data/mnist') #,one_hot=True)

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
train_data, train_labels, eval_data, eval_labels = load_data()
#train_data = train_data.reshape(train_data.shape[0], 1, 28, 28)
train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
train_labels = np_utils.to_categorical(train_labels, 10)

# Model mit Keras
model = Sequential()
model.add(InputLayer(input_shape=(28, 28, 1),name="Eingabe"))
model.add(BatchNormalization(name="Batch_Normalisierung"))

model.add(Conv2D(32,(5,5),padding="same",name="Conv2D_1",activation="relu"))
model.add(MaxPool2D(padding='same',name="Max_Pooling_1",pool_size=(2,2),strides=2))

model.add(Conv2D(64,(5,5),padding="same",name="Conv2D_2",activation="relu"))
model.add(MaxPool2D(padding='same',name="Max_Pooling_2",pool_size=(2,2),strides=2))
model.add(Flatten())

model.add(Dense(1024,activation='relu',kernel_initializer='random_uniform',name="Dense_fc_1"))
model.add(Dense(512,activation='relu',kernel_initializer='random_uniform',name="Dense_fc_2"))
model.add(Dense(10, activation='softmax',name="Ausgabe"))

model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adadelta(), metrics = ["accuracy","mse",metrics.categorical_accuracy])


# Wird aufgerufen, wenn das Training beginnt
def train_begin():
    url = 'http://localhost:9000/publish/train/begin' 
    post_fields = {"model":model.to_json()}     
    request = requests.post(url, data=post_fields)


lambda_cb = LambdaCallback(on_train_begin=train_begin())
remote_cb = RemoteMonitor(root='http://localhost:9000',path="/publish/epoch/end/",send_as_json=True)

#tf_cb = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

model.fit(train_data,train_labels, batch_size=64, epochs=100, verbose=1,validation_split=0.33,callbacks=[lambda_cb,remote_cb])
