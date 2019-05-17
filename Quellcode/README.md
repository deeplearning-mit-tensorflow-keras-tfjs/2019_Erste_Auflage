## 🖥Quellcode

Auf dieser Seite finden Sie den Quellcode zum Buch

## Kapitel 3: Neuronale Netze

Seite | Kurzbeschreibung | Dateipfad | Hinweise
------|------------------|-----------|---------
74|Klassifikation von Iris-Blüten mit scikit learn | [chap\_3/iris\_classification.py](chap\_3/iris\_classification.py)

## Kapitel 4: Python und Machine-Learning-Bibliotheken

Seite | Kurzbeschreibung | Dateipfad | Hinweise
------|------------------|-----------|---------
113 |Beispiele von Funktionalitäten von NumPy | [chap\_4/numpy\_examples.py](chap\_4/numpy\_examples.py) | Installieren Sie die zwei Python-Packages </br> `pip install tabulate wget`
117|Visualisierung vom Olivetti-Dataset|[chap\_4/olivetti\_dataset.py](chap\_4/olivetti\_dataset.py)
118|Normalisierung von Daten mit scikit-learn|[chap\_4/normalize\_iris\_dataset.py](chap\_4/normalize\_iris\_dataset.py)
122|Lineares Regressionsmodell mit scikit-learn|[chap\_4/linear\_regression.py](chap\_4/linear\_regression.py)

## Kapitel 5: TensorFlow

Seite | Kurzbeschreibung | Dateipfad | Hinweise
------|------------------|-----------|---------
136|Hello World in TensorFlow|[chap\_5/hello\_tensorflow.py](chap\_5/hello\_tensorflow.py)
137|Beispiele mit Tensoren|[chap\_5/tensors\_dimensions.py](chap\_5/tensors\_dimensions.py)
139|Bild in Tensoren laden|[chap\_5/loading\_picture\_in\_tensors.py](chap\_5/loading\_picture\_in\_tensors.py)
140|Beispiel der Verwendung von tf.Variable() und tf.assign()| [chap\_5/variables.py](chap\_5/variables.py)
142|Ein einfacher Graph mit TensorFlow|[chap\_5/graph.py](chap\_5/graph.py)
144|Graphenberechnung auf der GPU bzw. CPU| [chap\_5/graph\_gpu\_cpu.py](chap\_5/graph\_gpu\_cpu.py)
147|Graph mit Placeholders und Feed Dictionary|[chap\_5/graph\_with\_placeholder.py](chap\_5/graph\_with\_placeholder.py)
149|TensorBoard: Graph visualisieren|[chap\_5/my\_graph.py](chap\_5/my\_graph.py)
152|Graph in TensorBoard debuggen|[chap\_5/my\_graph\_tensorboard\_debug.py](chap\_5/my\_graph\_tensorboard\_debug.py)|Im Vorfeld TensorBoard starten mit : `tensorboard --logdir=logs --debugger\_port 12345`<br><br>⚠️ Unter Windows werden Sie leider diese Fehlermeldung bekommen, weil das Protokol zwischen TensorBoard und TensorFlow noch nicht implementiert ist `grpc:// debug URL scheme is not implemented on Windows yet.`<br>Siehe [https://github.com/tensorflow/tensorflow/issues/17933](https://github.com/tensorflow/tensorflow/issues/17933)
156|Projekt 1: Eine lineare Regression| [chap\_5/linear\_regression\_model.py](chap\_5/linear\_regression\_model.py)|
162|Projekt 2: Fashion MNIST Klassifikation mit TensorFlow und Modell speichern | [chap\_5/fashion-simple.py](chap\_5/fashion-simple.py)|
162|Fashion MNIST Klassifikation mit 2 verdeckten Schichten|[chap\_5/fashion-layers.py](chap\_5/fashion-layers.py)
171|TensorFlow-Modelle mit tf.train.Saver() speichern|[chap\_5/fashion\_with\_tf\_train\_saver.py](chap\_5/fashion\_with\_tf\_train\_saver.py)
174|TensorFlow-Modelle mit tf.saved\_model.builder.SavedModelBuilder() speichern|[chap\_5/fashion\_with\_saved\_model\_builder.py](chap\_5/fashion\_with\_saved\_model\_builder.py)
171-174|TensorFlow-Modelle laden mit tf.train.restore() und tf.saved\_model.loader.load()|[chap\_5/load\_fashion\_model.py](chap\_5/load\_fashion\_model.py)
177|Fashion MNIST Klassifikation mit CNNs|[chap\_5/fashion-cnn.py](chap\_5/fashion-cnn.py)|
184|Eager Execution| [chap\_5/eager.py](chap\_5/eager.py)

## Kapitel 6: Keras

Seite | Kurzbeschreibung | Dateipfad | Hinweise
------|------------------|-----------|---------
191|Version von Keras überprüfen|[chap\_6/keras\_version\_check.py](chap\_6/keras\_version\_check.py)
191|Benutzung der Sequential API|[chap\_6/keras\_xor\_sequential.py](chap\_6/keras\_xor\_sequential.py)|
192|Benutzung der Functional API|[chap\_6/keras\_xor\_functional.py](chap\_6/keras\_xor\_functional.py)|
196|Laden und speichern von Modellen| [chap\_6/keras\_load\_save.py](chap\_6/keras\_load\_save.py)|tfjs muss installiert werden, falls das Modell mit TensorFlow.js benutzt wird. `pip install tensorflowjs` und im Python-Code `import tensorflowjs as tfjs`
198|Keras Applications|[chap\_6/keras\_applications\_list.py](chap\_6/keras\_applications\_list.py)
198|Benutzung von Keras Applications|[chap\_6/keras\_applications\_list\_test.py](chap\_6/keras\_applications\_list\_test.py)
201|Klassifikation der Iris-Blumen ohne Evaluationsmetriken|[chap\_6/keras\_iris\_classification.py](chap\_6/keras\_iris\_classification.py)|Dataset: [chap\_6/data/](chap\_6/data/iris.csv)
201|Klassifikation der Iris-Blumen mit Evaluationsmetriken|[chap\_6/keras\_iris\_classification\_with\_evaluation.py](chap\_6/keras\_iris\_classification\_with\_evaluation.py)|Dataset: [chap\_6/data/](chap\_6/data/iris.csv)
204|Bildklassifikator auf Basis des CIFAR-10 Dataset|[chap\_6/keras\_cnn\_cifar.py](chap\_6/keras\_cnn\_cifar.py)|
204|Test des Bildklassifikators bzw. Laden und Benutzung eines gespeicherten Modells mit Keras|[chap\_6/keras\_cnn\_cifar\_test.py](chap\_6/keras\_cnn\_cifar\_test.py)
212|Aktienkursvorhersage|[chap\_6/keras\_stock\_prediction.py](chap\_6/keras\_stock\_prediction.py)|Dataset: [chap\_6/data/tsla.csv](chap\_6/data/tsla.csv)

## Kapitel 7: Netze und Metriken visualisieren

Seite | Kurzbeschreibung | Dateipfad | Hinweise
------|------------------|-----------|---------
217|TensorBoard|[chap\_7/tensorboard/simple\_conv\_net\_graph.py](chap\_7/tensorboard/simple\_conv\_net\_graph.py)|TensorBoard starten:</br>`tensorboard --logdir=logs/`
218|TensorBoard: Metriken visualisieren|[chap\_7/tensorboard/run\_metadata.py](chap\_7/tensorboard/run\_metadata.py)| 
223|TensorBoard: Benutzung des Text-Dashboard|[chap\_7/tensorboard/text\_summary\_reuters.py](chap\_7/tensorboard/text\_summary\_reuters.py)  
225|TensorBoard: Text-Dashboard mit tf.name\_scope()|[chap\_7/tensorboard/text\_summary\_with\_scope.py](chap\_7/tensorboard/text\_summary\_with\_scope.py)
226|TensorBoard: Benutzung von tf.summary.image()|[chap\_7/tensorboard/image\_summary\_cifar100.py](chap\_7/tensorboard/image\_summary\_cifar100.py)
228|TensorBoard: tf\_debug.TensorBoardDebugWrapperSession|[chap\_7/tensorboard/debugger\_wrapper\_session.py](chap\_7/tensorboard/debugger\_wrapper\_session.py)|Im Vorfeld TensorBoard starten: </br> `tensorboard --logdir=logs --debugger\_port 12345`</br></br>⚠️ Unter Windows werden Sie leider diese Fehlermeldung bekommen, weil das Protokol zwischen TensorBoard und TensorFlow noch nicht implementiert ist `grpc:// debug URL scheme is not implemented on Windows yet.` </br>Siehe https://github.com/tensorflow/tensorflow/issues/17933
232|Benutzung von tf\_cnnvis|[chap\_7/tensorboard/tf\_cnnvis\_example.py](chap\_7/tensorboard/tf\_cnnvis\_example.py)
234|Keras: Benutzung von plot\_model()|[chap\_7/plot\_model/plot\_model\_example.py](chap\_7/plot\_model/plot\_model\_example.py)|
237|Keras/Tensorboard: Benutzung vom TensorBoard Debugger|[chap\_7/tensorboard/keras\_tensorboard\_debugger.py](chap\_7/tensorboard/keras\_tensorboard\_debugger.py)|Im Vorfeld TensorBoard starten:`tensorboard --logdir=logs --debugger\_port 12345`<br><br>⚠️ Unter Windows werden Sie leider diese Fehlermeldung bekommen, weil das Protokol zwischen TensorBoard und TensorFlow noch nicht implementiert ist`grpc:// debug URL scheme is not implemented on Windows yet.` Siehe [https://github.com/tensorflow/tensorflow/issues/17933](https://github.com/tensorflow/tensorflow/issues/17933)</br></br>⚠️ Wenn Keras von Keras.io benutzt wird, kann `K.set_session()` benutzt werden. Wenn Keras von TensorFlow benutzt wird, muss folgende Zeile hinzugefügt werden: `import tensorflow.keras.backend as K`
238|Aktivierungen visualisieren|[chap\_7/activations/activations\_vis.py](chap\_7/activations/activations\_vis.py)|Wegen Bug in Keras von TensorFlow muss hier die Keras von keras.io benutzt werden 
242|Keras-Metriken mit Bokeh darstellen|[chap\_7/bokeh/keras\_history\_bokeh.py](chap\_7/bokeh/keras\_history\_bokeh.py)|Installieren Sie das Python Package *bokeh* </br> `pip install bokeh` 
244|Visualisierung von CNNs mit Quiver|[chap\_7/bokeh/quiver\_test.py](chap\_7/bokeh/keras\_history\_bokeh.py)|Bitte legen Sie einen leeren Ordner namens *tmp* im aktuellen Verzeichnis an
248|Projekt KeiVi|[chap\_7/keivi/](chap\_7/keivi/)|Installieren Sie die *node\_modules* für das Projet mit `npm install`
248|Benutzung von ConX|[chap\_7/conx/VGG\_19\_with\_ConX.ipynb](chap\_7/keivi/VGG\_19\_with\_ConX.ipynb)|Jupyter Notebook starten <br>`jupyter notebook` 

## Kapitel 8: TensorFlow.js

Starten Sie bitte vom Ordner *chap8* BrowserSync auf `browser-sync start --server --files "*.*"`
[http://localhost:3000](http://localhost:3000], um alle Beispiele zu testen<br>

Seite | Kurzbeschreibung | Dateipfad | Hinweise
------|------------------|-----------|---------
267-281|Operationen mit Tensoren|[chap\_8/](chap\_8/examples.html)
281|Quadratische Regression|[chap\_8/polynomial\_regression/](chap\_8/polynomial\_regression)
-|XOR-Modell mit TensorFlow.js|[chap\_8/](chap\_8/xor.html)
298|PoseNet-Modell|[chap\_8/posenet/](chap\_8/posenet)|Bitte passendes MP4-Video in den Ordner *./video* platzieren
298|Bildklassifikation mit ml5.js und MobileNet|[chap\_8/ml5\_js/](chap\_8/ml5\_js)
 
## Kapitel 9: Praxisbeispiele

Seite | Kurzbeschreibung | Dateipfad | Hinweise
------|------------------|-----------|---------
318|Projekt 1: Erkennung von Verkehrszeichen mit Keras|[chap\_9/1\_road\_signs\_keras](chap\_9/1\_road\_signs\_keras)|Bilder müssen in den Ordner *./img* platziert werden. Installieren Sie die das Python-Package webcolors `pip install webcolors`
333|Projekt 2: Intelligente Spurerkennung mit Keras|[chap\_9/2\_lane\_detection\_keras](chap\_9/2\_lane\_detection\_keras)|
334|Beispielcode zur Extraktion von einzelnen Frames mit OpenCV|[chap\_9/2\_lane\_detection\_keras/extract\_video\_frames.py](chap\_9/2\_lane\_detection\_keras/extract\_video\_frames.py)| Installieren Sie die das Python-Package cv2 `pip install opencv-python`
336|Visualisierung der Labels für das Dataset von Michael Virgo|[chap\_9/2\_lane\_detection\_keras/labels\_viewer.py](chap\_9/2\_lane\_detection\_keras/labels\_viewer.py)
343|Code für die Spurerkennung (Modell 1)|[chap\_9/2\_lane\_detection\_keras/lane\_detection.py](chap\_9/2\_lane\_detection\_keras/lane\_detection.py)
345|Code für die Spurerkennung (Modell 2) - basierend auf KITTI-Road Dataset|[chap\_9/2\_lane\_detection\_keras/train\_kitti\_road.py](chap\_9/2\_lane\_detection\_keras/train\_kitti\_road.py)|Das Training sollte wegen längeren Berechnungszeiten auf einem Rechner mit GPU durchgeführt werden. Sollte das nicht gegeben sein, kann das trainierte Modell [chap\_9/2\_lane\_detection\_keras/kitti\_road\_model.h5](chap\_9/2\_lane\_detection\_keras/kitti\_road\_model.h5) benutzt werden. </br></br>Im Vorfeld die Bilddateien von [http://www.cvlibs.net/download.php?file=data\_road.zip](http://www.cvlibs.net/download.php?file=data\_road.zip) herunterladen und nach Entzippen diese in den Ordner *./data* platzieren. </br></br>Adaptierter Code von: [https://github.com/6ixNugget/Multinet-Road-Segmentation]. </br></br>Bitte Dimensionen des Videobreiches anpassen, damit das das Dashcam Video kein Teil des Armaturenbretts beinhaltet (siehe Zeile 42)
345|Visualisierung (Modell 2)|[chap\_9/2\_lane\_detection\_keras/kitti\_road.py](chap\_9/2\_lane\_detection\_keras/kitti\_road.py)|Sowohl das fertige trainierte Modell [chap\_9/2\_lane\_detection\_keras/kitti\_road\_model.h5](chap\_9/2\_lane\_detection\_keras/kitti\_road\_model.h5) als auch die Videodatei *dash_cam.mp4* befinden sich in der ZIP-Datei von [https://www.rheinwerk-verlag.de/deep-learning-mit-tensorflow-keras-und-tensorflowjs_4715/](https://www.rheinwerk-verlag.de/deep-learning-mit-tensorflow-keras-und-tensorflowjs_4715/)
346|Projekt 3: YOLO und ml5.js|[chap\_9/3\_object\_detection\_yolo\_tfjs](chap\_9/3\_object\_detection\_yolo\_tfjs)|Bitte *index.html* anpassen, damit die passende MP4-Video Datei abgespielt wird
346|Projekt 4: VGG-19 mit Keras benutzen|[chap\_9/4\_vgg\_19\_keras](chap\_9/4\_vgg\_19\_keras)|Platzieren Sie Ihre Bilder in den Ordner *./samples* 
364|Projekt 5: Fashion-MNIST mit TensorFlow-Estimators|[chap\_9/5\_fashion\_mnist\_estimators\_tf](chap\_9/5\_fashion\_mnist\_estimators\_tf)|Platzieren Sie Ihre Bilder in den Ordner*./samples* 
364|Projekt 6: Stimmungsanalyse mit Keras|[chap\_9/6\_sentiment\_keras/sentiment.py](chap\_9/6\_sentiment\_keras/sentiment.py)|Installieren Sie den TensorFlow.js Konverter:`pip install tensorflowjs`
384|Projekt 7: Stimmungsanalyse mit TensorFlow.js|[chap\_9/7\_sentiment\_tfjs/](chap\_9/7\_sentiment\_tfjs)|Bitte den generierten Ordner */tfjs\_sentiment\_model* von 6\_sentiment\_keras kopieren, um das TensorFlow.js Modell benutzen zu können. 

## Kapitel 10: Ausblick
Seite | Kurzbeschreibung | Dateipfad | Hinweise
------|------------------|-----------|---------
318|AutoKeras|[chap\_10/auto\_keras.py](chap\_10/auto\_keras.py)|
