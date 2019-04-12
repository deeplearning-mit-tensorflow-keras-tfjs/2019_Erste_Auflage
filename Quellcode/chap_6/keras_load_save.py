#
# Modell trainieren, speichern der Modellstruktur und Parameter als JSON und .h5-Dateien mit Keras
# Trainiertes Modell samt Parameter werden mit Keras erneut geladen 
#

import tensorflow as tf

from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.models import Sequential, load_model,model_from_json
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras import metrics



from pprint import pprint
import numpy as np

# Einfaches Addieren 
input_data = np.array([
[	1	,	1	]	,
[	2	,	2	]	,
[	3	,	3	]	,
[	4	,	4	]	,
[	5	,	5	]])

output_data = np.array([[	2	],
[	4	],
[	6	],
[	8	],
[	10	]])

addition_model = Sequential()
addition_model.add(Dense(1024,input_dim=2,activation="linear"))
addition_model.add(Dense(1,activation="linear"))
addition_model.summary()

sgd = SGD(lr=0.001)
addition_model.compile(loss="mean_squared_error", optimizer=sgd,metrics=[metrics.mae])
addition_model.fit(input_data, output_data, batch_size=1, epochs=100, verbose=1)

# Modell wird gespeichert
addition_model.save("addition_model.h5")

# Und auch für TensorFlow.js!
# tfjs.converters.save_keras_model(addition_model, "./addition_model")

print("== Modell als JSON-Struktur ==")
pprint(addition_model.to_json())
pprint("== Modell als YAML-Struktur ==")
pprint(addition_model.to_yaml())

# Weights werden gespeichert
addition_model.save_weights("addition_weights.h5")

# Struktur des Modells wird als JSON gespeichert
json_str = addition_model.to_json()

with open("addition_model.json", "w") as json_file:
    json_file.write(json_str)

# Modell wird neu geladen (vom .h5 Datei)
model = load_model('addition_model.h5')
result = model.predict([[[5,5]]]) # Das Ergebnis müsste ungefähr bei 10 liegen
print("Ergebnis: {}".format(result))
del model

print("----------")

# Modell wird mit der Kombination JSON und weights geladen
with open('addition_model.json',"r") as f:
    json_file_content = f.read()
model = model_from_json(json_file_content) # bzw. model = model_from_yaml(json_file_content)
model.load_weights('addition_weights.h5')
result = model.predict([[[1,4]]]) # Das Ergebnis müsste ungefähr bei 5 liegen
print("Ergebnis: {}".format(result)) 


