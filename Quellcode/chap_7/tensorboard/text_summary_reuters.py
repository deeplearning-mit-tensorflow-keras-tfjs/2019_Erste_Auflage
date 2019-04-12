#
# Textausgabe im TensorBoard
# 	

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.datasets import cifar100
from tensorflow.python.keras.datasets import reuters

# Laden des Reuters-Datensatz
INDEX_FROM = 3
START_CHAR = 1
(x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz",
                                                         num_words=None,
                                                         skip_top=0,
                                                         maxlen=None,
                                                         test_split=0.2,
                                                         seed=113,
                                                         start_char=START_CHAR,
                                                         oov_char=2,
                                                         index_from=INDEX_FROM)

# Mapping Funktion von id auf Wort
word_index = reuters.get_word_index(path="reuters_word_index.json")
word_index = {k:(v+INDEX_FROM) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = START_CHAR # 1
word_index["<UNK>"] = 2
id_to_word = {value:key for key,value in word_index.items()}

# Funktion, die uns die Reuters Nachricht als String zurück gibt
def get_reuters_news(index):
    return ' '.join(id_to_word[id] for id in x_train[index] )

summary_writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph(),filename_suffix="reuters")

with tf.Session() as sess:
    for i in range (0,10):
        news = get_reuters_news(i)
        news_summary = tf.summary.text("News",tf.convert_to_tensor(news))
        summary_writer.add_summary(sess.run(news_summary),global_step=i)  