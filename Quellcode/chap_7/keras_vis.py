import tensorflow as tf
from tensorflow import keras
from keras.applications import VGG16
from keras.utils.vis_utils import plot_model
model = VGG16()
plot_model(model, to_file='model_output.png’, show_shapes=True, show_layer_names=True,rankdir="LR")
