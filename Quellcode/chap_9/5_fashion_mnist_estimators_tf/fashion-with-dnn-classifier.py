#
# Projekt 5 (Bonus): Fashion MNIST Klassifikationsaufgabe mit 
# TensorFlow und einem DNNClassifier Estimators

import tensorflow as tf
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt 

from PIL import Image
from skimage import color, exposure, transform, io
from tensorflow.examples.tutorials.mnist import input_data 
from sklearn.utils import shuffle

# Damit die textuelle Ausgaben vom Estimator während des Trainings
# sichtbar sind
tf.logging.set_verbosity(tf.logging.INFO)

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

#Feature Column
feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]

#Instanzierung eines DNNClassifiers
dnn_fashion_classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units = [512,256,16], # Hier werden die Anzahl der Neuronen pro verdeckten Schichten angelegt
    n_classes=10,
    model_dir="fashion_model_with_dnn_classifier")

# Training
def train_model():
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data}, #Bilder
        y=train_labels, # Labels 
        batch_size=64,
        num_epochs=None,
        shuffle=True)

    dnn_fashion_classifier.train(
        input_fn=train_input_fn,
        steps=20000)

# Evaluation
def evaluate_model():
    evaluate_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data}, #Bilder
        y=train_labels, # Labels 
        batch_size=64,
        #num_epochs=1,
        shuffle=True)
    accuracy_score = dnn_fashion_classifier.evaluate(input_fn=evaluate_input_fn)["accuracy"]
    print("Accuracy: {}%".format(accuracy_score*100))

#train_model()
evaluate_model()