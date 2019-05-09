## 📝Installationsanleitung

Zum schnellen Einstieg empfehlen wir die Python-Installation mit Anaconda: https://www.anaconda.com/distribution/

#### Organisation der Arbeitsumgebung
Wir empfehlen Ihnen ein Projektverzeichnis in Ihrem HOME-Verzeichnis anzulegen, etwa <i>deeplearning_buch</i>:
```mkdir deeplearning_buch```

Dann wechseln Sie zu diesem Verzeichnis und können dort die Beispiele vom Buch speichern.

#### Python-Installation mit Anaconda
Erzeugen Sie eine Umgebung namens <i>dl_env</i> mit der Python Version 3.6:
```conda create -n dl_env python=3.6```

Nachdem die Umgebung erzeugt wurde, muss sie nun aktiviert werden:
```source activate dl_env```

Um alle Beispiele der ersten Kapitel des Buches zu bearbeiten, empfehlen wir Ihnen in einer einzigen Aktion folgende Pakete zu installieren:

```conda install numpy scipy pandas scikit-learn matplotlib```
```conda install tensorflow```
```conda install keras```
```conda install tensorflowjs```

Ein Vorteil von conda ist es, dass beim Installieren eines Pakets nicht nur dieses sondern auch alle vom ihm benötigten Pakete mitinstalliert werden.


Wenn Sie die Liste alle definierten Umgebungen bekommen möchten, können Sie folgendes Kommando ausführen:
```conda env list```

#### Starten der Umgebung
Jedesmal, wenn Sie mit dem Buch arbeiten möchten, empfiehlt es sich zum Ihrer Arbeitsverzeichnis zu wechseln und die Python-Umgebung <i>dl_env</i> im Terminal zu aktivieren mit:
```source activate dl_env```

#### Beendigung der Arbeiten 
Nach Beendigung Ihrer Arbeiten sollten Sie die <i>dl_env</i> Umgebung deaktivieren:
```source deactivate dl_env```
