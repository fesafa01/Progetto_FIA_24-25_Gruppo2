import pandas as pd
import numpy as np

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
        '''
        Divide il dataset in training e test set.

        :param X: pandas DataFrame, features del dataset
        :param y: pandas Series, target del dataset

        :return: X_train, X_test, y_train, y_test
        '''

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

        return X_train, X_test, y_train, y_test
