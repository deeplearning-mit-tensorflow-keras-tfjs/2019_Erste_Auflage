#
# Graph mit TensorBoard debuggen
#

import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib
from tensorflow.python import debug as tf_debug

# Definition der Variablen 
a = tf.Variable(3.0,name="Variable_a")
b = tf.Variable(4.0,name="Variable_b")
c = tf.Variable(1.0,name="Variable_c")
d = tf.Variable(2.0,name="Variable_d")

# Initialisierung des Graph 
x_1 = tf.multiply(a,b,"Multiplikation")
x_2 = tf.add(c,d,"Addieren")
x_3 = tf.subtract(x_1,x_2,"Subtrahieren")
result = tf.sqrt(x_3,"Wurzel")

summary_writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph())
summary = tf.Summary()
summary.value.add(tag="Variable_a")

with tf.Session() as sess: 
    sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:12345")
    sess.run(tf.global_variables_initializer())
    res = sess.run(result)
    print("Ergebnis der Berechnung des Graphen:", res)
