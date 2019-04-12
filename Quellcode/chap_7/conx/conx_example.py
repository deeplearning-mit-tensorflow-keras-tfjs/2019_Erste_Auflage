# 
# Erstellung und Visualisierung eines Modelles mit ConX
#

import conx as cx

dataset = [[[0, 0], [0]],
           [[0, 1], [1]],
           [[1, 0], [1]],
           [[1, 1], [0]]]

net = cx.Network("XOR", 2, 5, 1, activation="sigmoid")
net.dataset.load(dataset)
net.compile(error='mean_squared_error',
            optimizer="sgd", lr=0.3, momentum=0.9)
net.train(2000, report_rate=10, accuracy=1.0)
net.picture()