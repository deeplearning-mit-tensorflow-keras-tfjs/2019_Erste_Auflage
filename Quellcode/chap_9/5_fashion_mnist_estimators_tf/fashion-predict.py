#
# Projekt 5: Beispiel einer Vorhersage mit einem 
# Estimator
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


fashion_classifier = tf.estimator.Estimator(model_dir="fashion_model",model_fn="fashion_model_fn")

## Vorhersage ##
def predict_image(picture):
    
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":picture},
        num_epochs=1,
        shuffle=False)
    
    fashion_predict = fashion_classifier.predict(input_fn=predict_input_fn)
    my_prediction = next(fashion_predict)
    predict_label = fashion_class_labels[np.argmax(my_prediction['probabilities'])]
    
    return predict_label



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
