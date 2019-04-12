#
# Textausgabe im TensorBoard
# 	


import tensorflow as tf
import numpy as np

summary_writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph(),filename_suffix="text_summary")

with tf.Session() as sess:
    with tf.name_scope('Meine_Ausgabe'):
            text_summary = tf.summary.text("Ausgabe 1",tf.convert_to_tensor("Hallo TensorFlow 😀"))
            summary_writer.add_summary(sess.run(text_summary),global_step=0)  
            text_summary = tf.summary.text("Tabelle",tf.convert_to_tensor("<table><thead><tr><th>Eine Tabelle </th></thead><tbody><tr><td>Eintrag 1</td></tr><tr><td>Eintrag 2</td></tr></tbody></table>"))
            summary_writer.add_summary(sess.run(text_summary),global_step=1)  
            text_summary = tf.summary.text("Link",tf.convert_to_tensor("<a href=\"http://www.tensorflow.org\">Link</a>"))
            summary_writer.add_summary(sess.run(text_summary),global_step=2)  
            text_summary = tf.summary.text("Liste",tf.convert_to_tensor("<ol><li>Item 1</li><li>Item 2</li></ol>"))
            summary_writer.add_summary(sess.run(text_summary),global_step=3)  