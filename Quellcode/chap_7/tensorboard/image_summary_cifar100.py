#
# Benutzung von tf.summary.image() 
# 

import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar100
from tensorflow.examples.tutorials.mnist import input_data 

# Laden des CIFAR-Datensatz
(cifar_images_train, labels_train), (images_test, labels_test) = cifar100.load_data(label_mode='fine')

# Laden des Fashion-Datensatz
fashion_images_train = input_data.read_data_sets('data/fashion',one_hot=True).train.images.reshape(-1, 28, 28, 1)

NUM_OF_IMAGES = 1000
# Ausgabe von N Bilder aus dem Datensatz
cifar_summary_writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph(),filename_suffix="cifar")
fashion_summary_writer = tf.summary.FileWriter("./logs",graph=tf.get_default_graph(),filename_suffix="fashion")
with tf.Session() as sess:
    # CIFAR
    cifar_image_summary = tf.summary.image("images", cifar_images_train[0:NUM_OF_IMAGES],max_outputs=NUM_OF_IMAGES,family="cifar")
    cifar_summary_writer.add_summary(sess.run(cifar_image_summary),0)
    # Fashion MNIST
    fashion_image_summary = tf.summary.image("images", fashion_images_train[0:NUM_OF_IMAGES],max_outputs=NUM_OF_IMAGES,family="fashion")
    fashion_summary_writer.add_summary(sess.run(fashion_image_summary),0)


