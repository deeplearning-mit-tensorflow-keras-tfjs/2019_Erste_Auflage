#
# Erstellung von einem Graph mit TensorFlow
#

import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib
from tensorflow.python import debug as tf_debug

# Initialisierung des Variablen 
a = tf.Variable(3.0,name="Variable_a")
b = tf.Variable(4.0,name="Variable_b")
c = tf.Variable(1.0,name="Variable_c")
d = tf.Variable(2.0,name="Variable_d")

# Initialisierung des Graph 
x_1 = tf.multiply(a,b,"Multiplikation")
x_2 = tf.add(c,d,"Addieren")
x_3 = tf.subtract(x_1,x_2,"Subtrahieren")
result = tf.sqrt(x_3,"Wurzel")

# Liefert Details zur GPU/CPU-Konfiguration des Rechners 
def get_GPU_CPU_details():
    print("GPU Vorhanden? ", tf.test.is_gpu_available())
    tf.test.gpu_device_name()
    print(device_lib.list_local_devices())

get_GPU_CPU_details()

# Graph mit GPU bzw. CPU
with tf.device("/gpu:0"): #Alternativ: /gpu:0
    with tf.Session() as sess:
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Wenn Sie mit dem TensorBoard arbeiten, können Sie die nächste Zeile auskommentieren
        # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:12345")

        sess.run(tf.global_variables_initializer())
        res = sess.run(result)
        print("Ergebnis der Berechnung des Graphs:",res)
