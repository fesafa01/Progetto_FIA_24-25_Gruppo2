import numpy as np
from classificatore_KNN import classificatore_KNN
from metrics_calculator import metrics_calculator

#Implementazione del metodo di validazione Holdout, che divide il dataset in due parti: training e test set
# in particolare, il test set è una porzione del dataset che non verrà utilizzata per l'addestramento del modello
# ma solo per la sua valutazione

class Holdout:
        
    def __init__(self, test_size=0.3):
        '''
        Costruttore della classe Holdout
        :param test_size: float, valore percentuale che rappresenta la 
        dimensione del test set rispetto all'intero set di dati, di default è 0.3
        in output non ritorna nulla

        '''
        if not 0 < test_size < 1:
            raise ValueError("test_size deve essere compreso tra 0 e 1.")
        self.test_size = test_size

    
    def splitter(self, X, y):
        """
        Suddivide il dataset in training e test set in base alla proporzione specificata.

        Il metodo esegue un mescolamento casuale degli indici del dataset, quindi divide i dati in due insiemi:
        - Training set: usato per addestrare il modello
        - Test set: usato per valutare le prestazioni del modello

        :param X: pandas DataFrame, rappresenta le feature del dataset.
        :param y: pandas Series, rappresenta le etichette di classe del dataset.

        :return:
            - X_train: subset del dataset usato per l'addestramento.
            - y_train: etichette corrispondenti al training set.
            - X_test: subset del dataset usato per il test.
            - y_test: etichette corrispondenti al test set.
        """

        # Controllo se X e y sono vuoti
        if X.empty or y.empty:
            raise ValueError("X e y non possono essere vuoti.")

        # Controllo che X e y abbiano la stessa lunghezza
        if len(X) != len(y):
            raise ValueError("X e y devono avere la stessa lunghezza.")

        # Shuffle degli indici
        indices = np.arange(len(X))  # Genera un array di indici numerici
        np.random.shuffle(indices)

        # Calcola il numero di campioni per il test set
        test_size = int(len(X) * self.test_size)

        # Divisione degli indici numerici in train e test
        train_indices = indices[test_size:]  # Indici per il training set
        test_indices = indices[:test_size]   # Indici per il test set

        # Suddivisione del dataset usando .iloc con indici numerici validi
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        return X_train, y_train, X_test, y_test

    def run(self, X,y,k=3):
        """
        Addestra e testa un classificatore KNN utilizzando la suddivisione del dataset in training e test set.
        
        :param X: pandas DataFrame, rappresenta le feature del dataset.
        :param y: pandas Series, rappresenta le etichette di classe del dataset.
        :param k: int, numero di vicini da considerare per il classificatore KNN.

        :return:
            - actual_value: etichette reali del test set.
            - predicted_value: valori predetti dal modello KNN sul test set.
        """
        
        
        # chiamiamo il metodo precedente per ottenere X_train, y_train, X_test, y_test
        X_train, y_train, X_test, y_test = self.splitter(X,y)

        # si crea il KNN con il parametro k specifico per l'iterazione a cui ci troviamo
        knn = classificatore_KNN(k)
            
        # si allena il modello sul training set specifico per l'iterazione a cui ci troviamo
        knn.fit(X_train, y_train)
            
        # si fa una predizione su y
        y_pred = knn.predict(X_test) 

        # si salvano in due variabili il valore corretto e quello predetto dal modello
        actual_value = y_test
        predicted_value = y_pred

        return actual_value, predicted_value

    def evaluate(self, X,y,k=3):
        """
        Valuta le prestazioni del classificatore KNN utilizzando le metriche di validazione.

        :param X: pandas DataFrame, rappresenta le feature del dataset.
        :param y: pandas Series, rappresenta le etichette di classe del dataset.
        :param k: int, numero di vicini da considerare per il classificatore KNN.

        :return:
            - metriche: dizionario contenente le metriche di valutazione del modello
        """
        
        # Eseguiamo il metodo run per ottenere i valori reali (actual) e predetti (predicted)
        actual_value, predicted_value = self.run(X,y,k)
        # creiamo un'istanza della classe che calcola le metriche di valutazione del modello
        Calculator = metrics_calculator()
        # calcoliamo la matrice di confusione
        matrix = Calculator.confusion_matrix(predicted_value, actual_value)
        # calcoliamo le metriche
        metriche = Calculator.metrics_evalutation(matrix, predicted_value, actual_value)
    
        return metriche