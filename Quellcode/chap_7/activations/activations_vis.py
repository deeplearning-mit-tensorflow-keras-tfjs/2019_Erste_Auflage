#
# Benutzung von keract zur Visualisierung der Aktivierungsschichten  
# Achtung: Wegen Bug in Keras TensorFlow muss hier auf die Keras.io
#
import tensorflow as tf
import numpy as np
import requests as requests
import keras 
import matplotlib.pyplot as plt 
import keract

from PIL import Image
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data 
from keras.layers import InputLayer, BatchNormalization, MaxPool2D, Conv2D,Flatten,Dense
from keras.models import Sequential
from keras import losses
from keras import optimizers
from keras import metrics
from keras.utils import np_utils
from keras.callbacks import LambdaCallback, RemoteMonitor
from keras.models import load_model, Model
from keract import get_activations


# Labels und Daten werden hier geladen
def load_data():
    data = input_data.read_data_sets('data/mnist'))

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
train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
train_labels = np_utils.to_categorical(train_labels, 10)
eval_labels = np_utils.to_categorical(eval_labels,10)

# Model mit Keras
def train_model():
        
   
    # Bug: https://github.com/keras-team/keras/issues/10417
    # model.add(InputLayer(input_shape=(1, 28, 28),name="Eingabe"))
    # model.add(BatchNormalization(name="Batch_Normalisierung"))
    model = Sequential()
    model.add(Conv2D(32,(5,5),padding="same",name="Conv2D_1",input_shape=(28, 28,1),activation="relu"))
    model.add(MaxPool2D(padding='same',name="Max_Pooling_1",pool_size=(2,2),strides=2))

    model.add(Conv2D(64,(5,5),padding="same",name="Conv2D_2",activation="relu"))
    model.add(MaxPool2D(padding='same',name="Max_Pooling_2",pool_size=(2,2),strides=2))
    model.add(Flatten())

    model.add(Dense(1024,activation='relu',kernel_initializer='random_uniform',name="Dense_fc_1"))
    model.add(Dense(512,activation='relu',kernel_initializer='random_uniform',name="Dense_fc_2"))
    model.add(Dense(10, activation='softmax',name="Ausgabe"))

    model.compile(loss=losses.categorical_crossentropy, 
    optimizer=optimizers.Adadelta(), metrics = ["accuracy","mse",metrics.categorical_accuracy])

    model.fit(train_data,train_labels, batch_size=64, epochs=2)
    model.save('my_model.h5')



def visualize_layers_output(model,layer_names,input,first_image):
    
    feature_maps = []
    fig = plt.figure(1)

    # Eingabebild wird dargestellt
    # Ggfs. für ein anderes Modell, 
    # muss dieser Block verändert werden
    ax = fig.add_subplot(len(layer_names)+1,1,1)
    ax.imshow(first_image,cmap=plt.gray())
    ax.set_axis_off()

    for i, layer_name in enumerate(layer_names):
        try:
            model.get_layer(layer_name)
        except ValueError as err:
            print("{} : Modell besitzt keinen Layer mit diesem Namen ".format(err))

        output_of_layer = model.get_layer(layer_name).output

        m = Model(inputs=model.input,outputs=output_of_layer)

        # Feature Map wird generiert
        feature_map = m.predict(input)

        ax = fig.add_subplot(len(layer_names)+1,1,i+2)
        
        # Überprüfung
        if len(feature_map.shape) == 4: # Wenn 4 dann ist es ein Bild (1,28,28,3)
            feature_maps.append(np.hstack(np.transpose(feature_map[0], (2, 0, 1))))
        else: 
            if len(feature_map.shape) == 2: # Ein 1D-Array 
                feature_maps.append(np.expand_dims(feature_map[0],0))
            
        ax.imshow(feature_maps[i],cmap='jet')
        ax.set_title(layer_name)
        ax.set_axis_off()
        del m
    
    plt.show()


# Optional: Training des Modells 
#train_model()

# oder direkte benutzung eines existierenden Modells
model = load_model("my_model.h5")
model.summary()

# Auswahl der Schichten des Modells für die Visualisierung
layers = ["Conv2D_1","Max_Pooling_1","Conv2D_2","Max_Pooling_2","Dense_fc_1","Dense_fc_2","Ausgabe"]
input = train_data[8:9]
visualize_layers_output(model,layers,input,first_image=img.reshape(28,28))


# Optionales Beispiel mit VGG16
'''
vgg = keras.applications.VGG16()
vgg.summary()
img_path = "./samples/cat.jpg"
img = np.array(Image.open(img_path).convert('RGB').resize((224, 224)))
input = np.expand_dims(img,axis=0) 
visualize_layers_output(vgg,["block1_conv2","block3_conv2"],input,first_image=img)
'''
