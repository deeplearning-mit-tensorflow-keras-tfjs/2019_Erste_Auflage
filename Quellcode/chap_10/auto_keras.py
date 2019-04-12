#
# Beispiel für die Benutzung von AutoKeras
#

from keras.datasets import mnist
from autokeras.image_supervised import ImageClassifier

# Wichtig: ansonsten RuntimeError
if __name__ == '__main__':
    # Laden der MNIST Daten
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    # Instanzierung des ImageClassifiers von AutoKeras
    clf = ImageClassifier(verbose=True,searcher_args={'trainer_args':{'max_iter_num':3}})

    # Ähnlich wie bei Keras, wird hier eine fit() Funktion benutzt. 
    # Der Parameter time_limit (in Sekunden) gibt an, wie lange AutoKeras nach den optimalen Modell suchen soll
    clf.fit(x_train, y_train, time_limit= 60*10)

    # Wenn ein Model gefunden wurde, wird es erneut trainiert
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)

    # Ähnlich Keras kann hier die predict()-Funktion aufgerufen werden
    results = clf.predict(x_test)
    print(results)
