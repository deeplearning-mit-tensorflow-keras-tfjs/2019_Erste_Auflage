# 
# Verwendung der Funktion plot_model() in Keras zur statischen Visualisierung eines Modelles
#

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.utils.vis_utils import plot_model
model = VGG16()
# Als PNG
plot_model(model, to_file='model_output.png', show_shapes=True, show_layer_names=True,rankdir="TB")
# Als SVG
plot_model(model, to_file='model_output.svg', show_shapes=False, show_layer_names=True,rankdir="TB")