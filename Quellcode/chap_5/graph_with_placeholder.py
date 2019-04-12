#
# Erstellung von einem Graph mit TensorFlow
#

import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib
from tensorflow.python import debug as tf_debug

# Initialisierung des Variablen 
a = tf.placeholder(tf.float16, name="Placeholder_a")
b = tf.placeholder(tf.float16, name="Placeholder_b")
c = tf.placeholder(tf.float16, name="Placeholder_c")
d = tf.placeholder(tf.float16, name="Placeholder_d")

# Initialisierung des Graph 
x_1 = tf.multiply(a, b, "Multiplikation")
x_2 = tf.add(c, d, "Addieren")
x_3 = tf.subtract(x_1, x_2, "Subtrahieren")
result = tf.sqrt(x_3, "Wurzel")


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(result, feed_dict={a:3.0,b:4.0,c:1.0,d:2.0})
    print("Ergebnis der Berechnung des Graphen:", res)
    sess.close()
