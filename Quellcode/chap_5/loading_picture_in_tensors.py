#
# Laden und Umkonvertierung eines RGB-Bildes zu einem Tensor mit Pillow und TensorFlow
#

import tensorflow as tf
import numpy as np
from PIL import Image


# Alternative 1: Mit Pillow
print ("== Alternative 1==")
with tf.Session() as sess:
    img = Image.open("cat.jpg" )
    img.load()
    # Umkonvertierung in numpy Array
    img_data = np.asarray( img, dtype="int32" )
    # Umkonvertierung als Tensor
    img_tensor = tf.convert_to_tensor(img_data, dtype=tf.int32)
    res = sess.run(img_tensor)
    print("Rank vom Tensor: {}".format(sess.run(tf.rank(img_data))))
  

# Alternative 2: Mit tf.image.decode_jpeg()
print ("== Alternative 2 == ")
with tf.Session() as sess:
    img = tf.image.decode_jpeg(
        tf.read_file("cat.jpg"))
    print("Rank vom Tensor: {}".format(sess.run(tf.rank(img))))
    print("Shape des Bildes: {}".format(sess.run(tf.shape(img))))
    img_content = img.eval()
    sess.run(img)
    print("RGB-Werte vom Pixel (x:0,y:0) :",img_content[0][0])
    print(img_content[0])

# Alternative 3: Mit tf.image.decode_image ohne Angaben von Bildformat()
print ("== Alternative 3 == ")
with tf.Session() as sess:
    img = tf.image.decode_image(
        tf.read_file("cat.jpg"))
    
    print("Rank vom Tensor: {}".format(sess.run(tf.rank(img))))
    print("Shape des Bildes: {}".format(sess.run(tf.shape(img))))
    img_content = img.eval()
    sess.run(img)
    print("RGB-Werte vom Pixel (x:0,y:0) :",img_content[0][0])

