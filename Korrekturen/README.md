## Korrekturen

Trotz sorgfältigem Redigieren und Prüfung lassen sich Fehler nie ganz vermeiden. Daher bietet wir Ihnen in diesem Dokument 
Korrekturen.
<br>
An dieser Stelle möchten wie Sie als Leser/Leserin unseres Buches dazu ermutigen, uns gefundene Fehler zu melden, so dass wir diese in besagtes Dokument aufnehmen und in eventuellen nächsten Auflagen berücksichtigen 
können. Vielen Dank dafür!

Seite | Kurzbeschreibung
------|------------------
143|Im Listing 5.1 : `init = tf.global_variables_initializer()`
147|aktualisierter Text: *<i>ersetzen Sie die tf.Variable() a, b, c und d durch tf.placeholder():</i>*  
157| Anpassung des Parameters *noise* in: `noise = np.random.randint(low=-5, high=5, size=input.shape)` Die Zeilen `import matplotlib.pyplot as plt` und `matplotlib.use('TkAgg')` müssen vertauscht werden
212|Aktualisierter Code: `rain_predict`→`train_predict`
370| Aktualisierter Code: `eval_metric_ops = {"accuracy": eval_accuracy } </br> print(eval_accuracy)`
376|Die Zeilen `num_epochs = None` und `num_epochs = 2` bitte löschen 
