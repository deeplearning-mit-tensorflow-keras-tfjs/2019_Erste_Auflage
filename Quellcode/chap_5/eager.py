#
#  Beispiel für die Benutzung des Eager Modus mit TensorFlow
#

import tensorflow as tf
tf.enable_eager_execution()
m = tf.add(2.0, 1)
m = tf.multiply(m,10)
m = tf.div(m,3)
print("Ergebnis (mit Eager Execution): {}".format(m))

