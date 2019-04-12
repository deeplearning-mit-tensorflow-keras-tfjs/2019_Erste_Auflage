#
# Projekt 5: Fashion MNIST Klassifikationsaufgabe mit 
# TensorFlow und Estimators
#

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

    #

    # Evaluationsdaten
    eval_data = data.test.images  
    eval_labels = data.test.labels
    eval_labels = np.asarray(data.test.labels,dtype=np.int32)
    eval_data, eval_labels = shuffle(eval_data, eval_labels)
    return (train_data, train_labels, eval_data, eval_labels) 

# Laden der Daten
train_data, train_labels, eval_data, eval_labels = load_fashion_data()

# Definition des Models 
def fashion_model_fn(features, labels, mode):
    
  print("######## Fashion Modell ##########")
  print("Modus: {}".format(mode))

  ####  Definition der Schichten ###
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  # Wir wollen 10 Klassen als Ergebnis 
  logits = tf.layers.dense(inputs=dropout, units=10)

  ####  Konfiguration des Estimators ###

  # Hier definieren wir die Vorhersage für den PREDICT and EVAL Mode
  predictions = {
     "classes": tf.argmax(input=logits, axis=1),
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    print("I'm predicting...")
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Berechnung des Loss (für TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # TRAIN Modus
  if mode == tf.estimator.ModeKeys.TRAIN:
    print("I'm training...")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())

    train_accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    tf.identity(train_accuracy[1],name="train_accuracy")
    tf.summary.scalar("train_accuracy",train_accuracy[1])

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Metriken für das EVAL modus
  if mode == tf.estimator.ModeKeys.EVAL:
    print("I'm evaluating...")

    eval_accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])

    eval_metric_ops =  {"accuracy":
        eval_accuracy
    }
    print(eval_accuracy)
    print(eval_accuracy[1])
    tf.identity(eval_accuracy[1],name="eval_accuracy")
    tf.summary.scalar("eval_accuracy",eval_accuracy[1])

    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
#Generiere eine Estimator bzw. Klassifier
fashion_classifier = tf.estimator.Estimator(model_fn=fashion_model_fn,model_dir="fashion_model")

# Trainingsfunktion
def train_model():
 
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    #tensors_to_log = {"probabilities": "softmax_tensor","train_accuracy"}
    logging_hook = tf.train.LoggingTensorHook(
    {"train_accuracy"}, every_n_iter=50)

    # Training fängt hier 
    fashion_classifier.train(
        input_fn=train_input_fn,
        hooks=[logging_hook],
        steps=20000)

# Evaluationsfunktion
def eval_model():
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    logging_hook = tf.train.LoggingTensorHook({"eval_accuracy"}, every_n_iter=10)

     # Evaluation fängt hier 
    return fashion_classifier.evaluate(
        input_fn=eval_input_fn,
         hooks=[logging_hook],
        steps=2000)

## Vorhersage ##
def predict_image(picture):
    
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":picture},
        #num_epochs=1,
        shuffle=False)
    
    fashion_predict = fashion_classifier.predict(input_fn=predict_input_fn)
    my_prediction = next(fashion_predict)
    predict_label = fashion_class_labels[np.argmax(my_prediction['probabilities'])]
    
    return predict_label


train_model()
eval_model()


### Variante 1 mit PIL ####
image = Image.open("samples/coat.jpg")
grayscale_image = image.convert('L').resize((28,28))
input_image = []
for x in range(0,28):
    for y in range(0,28):
        currentPixel = grayscale_image.getpixel((y,x))
        input_image.append(currentPixel)
        
input_image = np.array(input_image,dtype="float32")
input_image = input_image.reshape((28,28))
input_image = - (input_image - 255.0 ) / 255.0


plt.imshow(image)
title = "Prediction: {}".format(predict_image(input_image))
plt.title(title)
plt.show()



'''
# Variante 2 mit OpenCV ! 
image = cv2.imread('samples/shoe2.jpg')
grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
input_image = cv2.resize(grayscale_image,(28,28)).reshape((28,28))
input_image = - (input_image - 255.0 ) / 255.0
input_image = np.array(input_image,dtype="float32")

title = "Prediction: {}".format(predict_image(input_image))
plt.title(title)
plt.imshow(image)
plt.show()
'''


