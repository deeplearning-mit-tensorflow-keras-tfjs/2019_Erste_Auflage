## 📝Korrekturen

Trotz sorgfältigem Redigieren und Prüfung lassen sich Fehler nie ganz vermeiden. Daher bieten wir Ihnen in diesem Dokument 
Korrekturen.
<br>
An dieser Stelle möchten wie Sie als Leser/Leserin unseres Buches dazu ermutigen, uns über den Verlag gefundene Fehler zu melden, so dass wir diese in dieses Dokument aufnehmen und in eventuellen nächsten Auflagen berücksichtigen 
können. Vielen Dank dafür!

#### Seite 59 (Fußnote 3)

"den wir es später betrachten werden"

ersetzen durch

"den wir erst später betrachten werden"


*(Dank an einen aufmerksamen Leser)*

#### Seite 60

"anschließende Aktivierungsfunktion 𝜑(x)"

ersetzen durch

"anschließende Aktivierungsfunktion 𝜑(𝛼)"


*(Dank an den aufmerksamen Leser aus Hamburg)*

#### Seite 70

"cross entropy"

ersetzen durch

"cross validation" 


#### Seite 72

Hinweis: statt der Datei https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data, die kein Header für die Spaltennamen (*<i>sepal length,sepal width,petal length,petal width,species</i>*) beinhaltet, sollte die Datei iris.csv benutzt werden. 

*(Dank an Christian)*

#### Seite 74

"data_train_array = data_train.as_matrix()"

ersetzen durch

"data_train_array = data_train.values"
oder
"data_train_array = data_train.to_numpy()"

*(Dank an Leonhard)*

#### Seite 78

"accuracy = (TP + FN) / (TP + FP + TN + FN)"

ersetzen durch

"accuracy = (TP + TN) / (TP + FP + TN + FN)"

*(Dank an einen aufmerksamen Leser)*

#### Seite 89

"werden Sie bestimmt früh oder später damit konfrontiert sein."

ersetzen durch

"werdenSie bestimmt früher oder später damit konfrontiert sein."


*(Dank an einen aufmerksamen Leser)*

#### Seite 90

"denken Sie an einem Sensor"

ersetzen durch

"denken Sie an einen Sensor"


*(Dank an einen aufmerksamen Leser)*

#### Seite 138

Nach dem Satz "Tensoren können ebenfalls mit der Python-Bibliothek NumPy initialisiert werden:"
```python 
import tensorflow as tf
import numpy as np

np_array = np.arange(0,5,step=0.5)
tensor_from_numpy = tf.constant(np_array,dtype=tf.float16)
print(sess.run(tensor_from_numpy))
```

ersetzen durch

```python 
import tensorflow as tf
import numpy as np

with tf.Session()as sess:
    np_array = np.arange(0,5,step=0.5)
    tensor_from_numpy = tf.constant(np_array,dtype=tf.float16)
    print(sess.run(tensor_from_numpy))
``` 


*(Dank an Tician)*

#### Seite 143 

Im Listing 5.1 : 
```python 
init = tf.global_variables_initializer()
``` 

#### Seite 147 
Ersetzen Sie ```tf.constant()``` durch``` tf.Variable()```in dem Textausschnitt: 
<i>Nehmen Sie unseren Beispielgraphen aus Abbildung 5.11, und ersetzen Sie die tf.Variable() a, b, c und d durch tf.placeholder():</i>
 

#### Seite 157 

Anpassung des Parameters *noise* in: 

```python 
noise = np.random.randint(low=-5, high=5, size=input.shape)
``` 

Folgende Zeilen vertauschen:
```python 
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') 
``` 

#### Seite 160

Um den Graphen vom Block 5.6.4 darzustellen, vergewissern Sie sich, dass Sie nach der Zeile 

```python 
print("Vorhersage Wert für w: " + str(weight_value))   
``` 
folgendes hinzufügen:
``` 
plt.show(block=True)
```  
*(Dank an Christian)*

#### Seite 178 
Die Zeilen ```train_labels = data.train.labels``` und ```eval_labels = data.train.labels``` sind überflüssig und sollen ignoriert werden.

*(Dank an den/die Amazon Rezensenten/Rezensentin)*

#### Seite 196 
```evaluation_results = model.evaluate(input_test_data, input_test_data)``` 

ersetzen durch 

```evaluation_results = model.evaluate(input_test_data, output_test_data)``` 


#### Seite 212

Aktualisierter Code: `train_redict` → `train_predict`

#### Seite 368

Aktualisierter Code:

```python 
eval_metric_ops = {"accuracy": eval_accuracy} 
print(eval_accuracy)
```

#### Seite 376

Folgende Zeilen löschen: 
```python 
num_epochs = None
``` 
```python 

num_epochs = 2
``` 

