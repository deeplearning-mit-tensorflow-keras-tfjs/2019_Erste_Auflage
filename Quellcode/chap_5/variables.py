#
# Beispiel der Verwendung von tf.Variable() und tf.assign()
#

import tensorflow as tf
import numpy as np

my_tensor = tf.Variable([[1, 2, 3], [4,5,6]],name="meine_variable")

# Erstes Beispiel
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t = np.array([2]) 

    # Äquivalent zu my_tensor = my_tensor * 2 
    my_tensor = my_tensor * t
    my_tensor = tf.square(my_tensor) / 4
    my_tensor = my_tensor - 1
    print("Ausgabe Beispiel 1:")
    print(sess.run(my_tensor))

    # Ausgabe:
    # [[ 0.  3.  8.]
    # [15. 24. 35.]]


# Zweites Beispiel:
my_tensor = tf.Variable([[1, 2, 3], [4,5,6]],name="meine_variable")
my_second_tensor = tf.assign(my_tensor,[[0,0,7],[0,0,8]])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    my_tensor = my_second_tensor
    print("Ausgabe Beispiel 2:")
    print(sess.run(my_tensor))
    
    # Ausgabe:
    # [[0 0 7]
    # [0 0 8]]
