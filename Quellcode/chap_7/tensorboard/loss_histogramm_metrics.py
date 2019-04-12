


import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data 
import tensorboard.plugins.beholder as beholder_lib
from tf_cnnvis import *
from tensorflow.python import debug as tf_debug 

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

# Labels und Daten werden hier geladen
def load_fashion_data():
    data = input_data.read_data_sets('data/fashion',one_hot=True)

    # Trainingsdaten
    train_data = data.train.images


    train_labels = data.train.labels
    train_labels = np.asarray(data.train.labels,dtype=np.int32)
    
    # Evaluationsdaten
    eval_data = data.test.images  
    eval_labels = data.test.labels
    eval_labels = np.asarray(data.test.labels,dtype=np.int32)
    eval_data, eval_labels = shuffle(eval_data, eval_labels)
    return (train_data, train_labels, eval_data, eval_labels) 

# Laden der Daten
train_data, train_labels, eval_data, eval_labels = load_fashion_data()
train_data = train_data.reshape(-1, 28, 28, 1)

# Placeholders für die Bilder und die Labels
images = tf.placeholder("float", [None, 28,28,1],"images")
labels = tf.placeholder("float", [None, 10],"labels")

global conv1

####  Definition der Schichten ###
def build_fashion_model(input, labels):

  # Input Layer
  input_layer = tf.reshape(images, [-1, 28, 28, 1])
 
  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters= 32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu, name="1_layer")

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2,name="1_maxpooling")
    
  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,name="2_layer")
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2,name="2_max_pooling")

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu,name="3_dense")
  dropout = tf.layers.dropout(inputs=dense, rate=0.4)
  logits = tf.layers.dense(inputs=dropout, units=10,name="4_logits")

  return logits, input_layer


pred, img = build_fashion_model(images,labels)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=labels))
train_op = tf.train.AdamOptimizer(0.001).minimize(loss_op) 



EPOCHS_NUM = 1
epochs = range(0,EPOCHS_NUM)

BATCH_SIZE = 1


batches = range(0,len(train_labels)//BATCH_SIZE)

# 
def get_next_batch(data,batch):
    batch_start_index = batch * BATCH_SIZE
    batch_end_index = min((batch+1)*BATCH_SIZE, len(data))
    return data[batch_start_index:batch_end_index]

summary_writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph(),filename_suffix="fashion")
#beholder_hook = beholder_lib.BeholderHook("./logs")

beholder = beholder_lib.Beholder(logdir="./logs")


# Wir gruppieren die Summary zusammen
with tf.name_scope('accuracy_loss_metrics'):
    accuracy_summary = tf.summary.scalar('accuracy', accuracy_op)
    loss_summary = tf.summary.scalar('loss', loss_op)


# Datenset visualisieren: 20 erste Bilder des Datensets 
input_image_summary = tf.summary.image("images", train_data[0:20],max_outputs=20)


# Labels als String für das Text Dashboard in TensorBoard
train_labels_as_string = np.array2string(train_labels,max_line_width=10)
train_labels_as_string = train_labels_as_string.replace("[","\n[")

#input_labels_summary = tf.summary.text("labels",tf.convert_to_tensor(train_labels_as_string))

# Histogramm
loss_summary_histo = tf.summary.histogram('loss', loss_op)

# Hilfefunktion, um die Summaries zum writer hinzuzufügen
def add_summaries_to_tensorboard(summaries,step):
    for summary in summaries:
        summary_writer.add_summary(summary,global_step=step)

step_var = tf.Variable(0.0)
             
# Launch the graph in a session
with tf.Session() as sess:
#with tf.train.MonitoredSession(hooks=[beholder_hook]) as sess:
    sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:12345")

     # Muss weg, wenn mit MonitoredSession 
    sess.run(tf.global_variables_initializer())
    with tf.name_scope('Mein_Step_Counter'):
        counter_summary = tf.summary.text("Step",tf.convert_to_tensor("Ich bin step: {}".format(step_var.eval())))
 
   
    summary_writer.add_summary(sess.run(input_image_summary),0)  

    steps = 0

    for i in epochs:
        for b in batches:
           
            # Batch für die Features
            img_input = get_next_batch(train_data,b)
            
            # Batch für die Labels
            labels_output = get_next_batch(train_labels,b)
            
            predicted_output = sess.run(train_op,feed_dict={images:img_input,labels:labels_output})



            loss, acc = sess.run([loss_op,accuracy_op],feed_dict={images:img_input,labels:labels_output})

            # Für die Summary
            summaries = sess.run([accuracy_summary,loss_summary],feed_dict={images:img_input,labels:labels_output})
            add_summaries_to_tensorboard(summaries,steps)

            loss_histogramm = sess.run(loss_summary_histo,feed_dict={images:img_input,labels:labels_output})
            summary_writer.add_summary(loss_histogramm,steps)  

            tf.add(step_var,1.0)
            summary_writer.add_summary(sess.run(counter_summary),steps)  
            
            steps = steps + 1


            # Beholder
            beholder.update(session=sess,frame=img_input[0])

            feed_dict={images:img_input,labels:labels_output}

            print("Iter " + str(i) + ", Loss= " + \
                            "{:.6f}".format(loss) + ", Training Accuracy= " + \
                            "{:.5f}".format(acc))

        # deconv visualization
    layers = ["r", "p", "c"]
    total_time = 0

        # api call
    '''is_success = deconv_visualization(sess_graph_path = sess, value_feed_dict = feed_dict, 
                                    input_tensor=img, layers=layers, 
                                    path_logdir="./logs", 
                                    path_outdir="./output")
    '''
    is_success = activation_visualization(sess_graph_path = sess, value_feed_dict = feed_dict, 
                                    input_tensor=img, layers=layers, 
                                    path_logdir="./logs", 
                                    path_outdir="./output")