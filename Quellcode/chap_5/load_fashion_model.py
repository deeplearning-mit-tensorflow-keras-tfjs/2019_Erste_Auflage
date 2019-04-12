#
# Laden von einem Modell (fashion_model) mit TensorFlow 
# 

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 
from PIL import Image
import random
import pprint as pprint
from tensorflow.examples.tutorials.mnist import input_data 

#tf.logging.set_verbosity(tf.logging.DEBUG)
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

# Mit dem Parameter one_hot  
# wird eine Zahl als Array mit 10 Elementen transformiert z.B. 8 wird als
# [0 0 0 0 0 0 1 0] kodiert

# Lädt die labels und Daten
def load_fashion_data():
    data = input_data.read_data_sets('data/fashion',one_hot=True)

    # Trainings daten
    train_data = data.train.images
    train_labels = data.train.labels
    
    # Evaluations daten
    eval_data = data.test.images  
    eval_labels = data.test.labels
    eval_data, eval_labels = shuffle(eval_data, eval_labels)
    return (train_data, train_labels, eval_data, eval_labels) 


train_data, train_labels, eval_data, eval_labels = load_fashion_data()
input_label = fashion_class_labels[np.argmax(eval_labels[0], axis=None, out=None)]
input_image = [eval_data[0]]

print("Ausgewähltes Bild: {}".format(input_label))




# Variante 1: Laden mit tf.train.restore()
def with_model_saver():

    with tf.Session() as sess:
    
        # Lädt das Model bzw den Graph 
        model_saver = tf.train.import_meta_graph('./fashion_model.ckpt.meta')
        model_saver.restore(sess,tf.train.latest_checkpoint('.'))
    
        # Unser Graph ist nun geladen
        current_graph = tf.get_default_graph()

        # Listet alle Operationen auf, die beim restaurieretn Graph 
        # gespeichert wurde
        # print(current_graph.get_operations())
        # pprint(current_graph.get_operations())

        # Hier brauchen wir zwei Tensoren zum "füttern" des Modells
        # Bei der Operation get_tensor_by_name() benutzt Tensorflow als Eingabe das Format <name_des_tensors:index> bzw. <name_der_operation:index>
        # Wenn nur ein Tensor von der Operation ausgegeben wird das index 0 angegeben 
        X = current_graph.get_tensor_by_name("X:0")


        # Wichtig: Damit diese Funktion erfolgreich aufgerufen wird, muss bei fashion_with_tf_train_saver.py
        # die Funktion fashion_model umgeschrieben werden:
        # def fashion_model(weights,bias,X):
        #    return tf.add(tf.matmul(X,weights),bias,"fashion_model")
        
        restored_fashion_model = current_graph.get_tensor_by_name("fashion_model:0")


        # Das Model ist nun als Tensor geladen worden
        # Wir können nun unser model wieder benutzen:
        predictions = sess.run(restored_fashion_model,feed_dict={X:input_image})
        index = int(np.argmax(predictions,axis=1))
        
        # Und hier die Vorhersage:
        print("Gefundene Fashion Kategorie: {}".format(fashion_class_labels[index]))



# Variante 2 mit saved_model.loader

def with_saved_model():
     with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ["train"], "./model_export")
        current_graph = tf.get_default_graph()
        # print(current_graph.get_operations())
        X = current_graph.get_tensor_by_name("X:0")

        restored_fashion_model = current_graph.get_tensor_by_name("fashion_model:0")

        # Das Model ist nun als Tensor geladen worden
        # Wir können nun unser model wieder benutzen:
        predictions = sess.run(restored_fashion_model,feed_dict={X:input_image})
        index = int(np.argmax(predictions,axis=1))
        
        # Und hier die Vorhersage:
        print("Gefundene Fashion Kategorie: {}".format(fashion_class_labels[index]))




# Laden der Modelle 
#with_model_saver()
with_saved_model()
