# 
# Keras Metriken mit Bokeh darstellen 
#

import tensorflow as tf
from tensorflow import keras
from keras import *
from keras.layers import *
import numpy as np
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data 
from keras.utils import np_utils
from keras.initializers import Constant
from keras import metrics
from tensorflow.python import debug as tf_debug
import keras
from bokeh.plotting import figure, output_file, show

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
    data = input_data.read_data_sets('data/fashion') 

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

print(train_data.shape)

# Model mit Keras
model.add(InputLayer(input_shape=(28, 28,1),name="1_Eingabe"))
model.add(Conv2D(32,(2, 2),padding='same',bias_initializer=Constant(0.01),kernel_initializer='random_uniform',name="2_Conv2D"))
model.add(Activation(activation='relu',name="3_ReLu"))
model.add(MaxPool2D(padding='same',name="4_MaxPooling2D"))
model.add(Conv2D(32,(2, 2),padding='same',bias_initializer=Constant(0.01),kernel_initializer='random_uniform',name="5_Conv2D"))
model.add(Activation(activation='relu',name="6_ReLu"))
model.add(MaxPool2D(padding='same',name="7_MaxPooling2D"))
model.add(Flatten())
model.add(Dense(1024,activation='relu',bias_initializer=Constant(0.01),kernel_initializer='random_uniform',name="8_Dense"))
model.add(Dropout(0.4,name="9_Dense"))
model.add(Dense(10, activation='softmax',name="10_Ausgabe"))

model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adadelta(), metrics = ["accuracy","mse",metrics.categorical_accuracy])

model_history = model.fit(train_data,train_labels, batch_size=64, epochs=100, verbose=1,validation_split=0.5)

# Liste alle verfügbaren History
print(model_history.history.keys())



## Benutzung von Bokeh ##
output_file("keras_metrics.html")

p = figure(title="Keras Metriken",plot_width=1200, plot_height=400,tools="pan,wheel_zoom,hover,reset",sizing_mode="scale_both")

# Loss 
x_train_loss_axis = np.arange(0,len(model_history.history['loss'])) 
y_train_loss_axis = model_history.history['loss']

# Loss (validation)
x_validation_loss_axis = np.arange(0,len(model_history.history['val_loss'])) 
y_validation_loss_axis = model_history.history['val_loss']

# Accuracy
x_train_acc_axis = np.arange(0,len(model_history.history['val_acc']))
y_train_acc_axis = model_history.history['val_acc']

x_validation_acc_axis = np.arange(0,len(model_history.history['val_acc']))
y_validation_acc_axis = model_history.history['acc']

# Labels für die jeweilige Axis
p.xaxis.axis_label = "Epochs"
p.yaxis.axis_label = "Wert"

# Kurve werden generiert
p.line(x_train_loss_axis,y_train_loss_axis,legend="Loss / Training",line_color="red",line_width=2,alpha=0.5)
p.line(x_validation_loss_axis,y_validation_loss_axis,legend="Loss / Validation",line_color="red",line_width=2)

p.line(x_train_acc_axis,y_train_acc_axis,legend="Accuracy / Training",line_color="green",line_width=2,alpha=0.5)
p.line(x_validation_acc_axis,y_validation_acc_axis,legend="Accuracy / Validation",line_color="green",line_width=2)

# Die HTML-Datei wird generiert
show(p)
