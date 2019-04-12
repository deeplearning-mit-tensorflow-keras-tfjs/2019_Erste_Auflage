#
# Verwendung der Keras Applications
#

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.applications.densenet import DenseNet201
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from  tensorflow.python.keras.models import Model

# Für jedes Modell wird die Summary ausgegeben 

print("=== VGG 16 ===")
VGG16().summary()

print("=== VGG 19 ===")
VGG19().summary()

print("=== ResNet50 ===")
ResNet50().summary()

print("=== DenseNet121 ===")
DenseNet121().summary()

print("=== DenseNet201 ===")
DenseNet201().summary()

print("=== InceptionResNetV2 ===")
InceptionResNetV2().summary()
