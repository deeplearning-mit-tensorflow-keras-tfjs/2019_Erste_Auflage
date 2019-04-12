#
# Benutzung von Tensoren mit TensorFlow
#

import tensorflow as tf
import numpy as np

with tf.Session() as sess:

    my_scalar = tf.constant(1,name="mein_Skalar")
    print(my_scalar) 
    
    #Ausgabe: Tensor("Const_2:0", shape=(1, 1), dtype=int32)
    tensor_0d = tf.constant(3)
    print(sess.run(tensor_0d))

    tensor_1d = tf.constant([1,2,3,4])
    print(sess.run(tensor_1d))

    tensor_1d_with_strings = tf.constant(["Hallo","Welt","dies","ist","ein 1D Tensor!"])
    print(sess.run(tensor_1d_with_strings))

    tensor_2d_with_strings = tf.constant([["Petra","Schmitt"],["Max","Mustermann"],["John","Doe"]])
    print(sess.run(tensor_2d_with_strings))

    tensor_2d_with_integer = tf.constant([[1,2],[3,4],[5,6]])
    print(sess.run(tensor_2d_with_integer))

    tensor_3d = tf.constant([
                               [[1,4],[3,8]],
                               [[5,7],[9,3]]
                            ])
    print(sess.run(tf.rank(tensor_3d)))

    np_array = np.arange(0,5,step=0.5)
    tensor_from_numpy = tf.constant(np_array,dtype=tf.float16)
    #[0.  0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5]
    print(sess.run(tensor_from_numpy))