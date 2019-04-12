#
# Fashion MNIST Klassifikation mit TensorFlow 
# und speichern des trainierten Modells mit tf.train.Saver()
# 
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt 

from PIL import Image
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data 

tf.logging.set_verbosity(tf.logging.DEBUG)
# tf.logging.set_verbosity(tf.logging.INFO)

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
    
    # Wir benutzen die shuffle-Funktion, damit die Daten durchgemischt werden
    # und zufällig ausgewählt werden.
    
    eval_data, eval_labels = shuffle(eval_data, eval_labels)
    return (train_data, train_labels, eval_data, eval_labels) 

# Definition des Modells
def fashion_model(weights,bias,X):
    return tf.add(tf.matmul(X,weights),bias,"fashion_model")


train_data, train_labels, eval_data, eval_labels = load_fashion_data()

# Variablen
dimension = 784 # 28x28 Pixels
number_of_classes = 10 

# Eingabe (Pixels)
X = tf.placeholder(tf.float32,[None,dimension],name="X")

# Ausgabe (Klasse bzw. Kategorie)
Y = tf.placeholder(tf.float32,[None,number_of_classes],name="Y")

# Gewichtungen
w = tf.Variable(tf.random_normal([dimension,number_of_classes],stddev=0.01),name="weights",dtype=tf.float32)

# Bias
b = tf.Variable(tf.random_normal([number_of_classes]), name="bias",dtype=tf.float32)

predict_input = tf.placeholder(tf.float32,[None,dimension],name="predict_input")

# Aufruf des Modells
model_out = fashion_model(w,b,X)

# Optionale Ausgabe eines Bildes aus dem Fashion MNIST 
input_picture = eval_data[0].reshape(28,28)
input_label = fashion_class_labels[np.argmax(eval_labels[0], axis=None, out=None)]
print ("Ausgewählt: {}".format(input_label))
plt.title(input_label)
plt.imshow(input_picture,cmap='Greys')

# Auskommentieren
plt.show()

# Definition der Kosten/Loss-Funktion
# 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_out,labels=Y))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) 

epochs = range(0,1000)

# Trainingsschleife
with tf.Session() as sess:
    saver = tf.train.Saver() 
    sess.run(tf.global_variables_initializer())
    for i in epochs:
        sess.run(train_op, feed_dict={X: train_data, Y: train_labels})
        loss = sess.run(cost,feed_dict={X: train_data, Y: train_labels})
        accuracy = np.mean(np.argmax(sess.run(model_out,feed_dict={X: train_data, Y: train_labels}),axis=1) == np.argmax(train_labels,axis=1))
        
        if(i%10==0):
            print("Epoch {} // Accuracy: {} Loss:{}".format(i,accuracy,loss))

    # Ausgabe der Vorhersage für das erste Element von eval_data
    predictions = sess.run(model_out,feed_dict={X:[eval_data[0]]})
    index = int(np.argmax(predictions,axis=1))
    print("Gefundene Fashion Kategorie: {}".format(fashion_class_labels[index]))

    saver.save(sess,"./fashion_model.ckpt")
    print("Modell wurde gespeichert")