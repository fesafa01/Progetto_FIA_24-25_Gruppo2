import pandas as pd
import numpy as np
from classificatore_KNN import classificatore_KNN

"""
Questo modulo implementa il metodo di Random Subsampling per la valutazione di un modello di classificazione.

Il Random Subsampling consiste nel suddividere ripetutamente un dataset in training e test set in modo casuale,
eseguendo un numero predefinito di iterazioni. 
In ogni iterazione, il modello viene addestrato e testato,
e la prestazione viene misurata per ottenere una valutazione complessiva (data dalla media delle prestazioni singole).
"""

class RandomSubsampling:
    
    def __init__(self, test_size=0.3, num_splits=10):
        """
        Metodo costruttore per il random subsampling.
        
        test_size: percentuale del dataset usata per il test set.
        num_splits: numero di iterazioni per la suddivisione casuale.
        """

        # Controllo della validità di test_size
        if not 0 < test_size < 1:
            raise ValueError("test_size deve essere compreso tra 0 e 1.")
        self.test_size = test_size
        self.num_splits = num_splits

    def splitter(self, X, y):
        """
        Divide il dataset in training e test set num_splits volte in modo casuale.
        
        :param X: pandas DataFrame, features del dataset
        :param y: pandas Series, target del dataset
        
        :return: lista di tuple (X_train, X_test, y_train, y_test) per ogni split
        """

        # Controllo che X e y non siano vuoti
        if X.empty or y.empty:
            raise ValueError("X e y non possono essere vuoti.")

        if len(X) != len(y):
            raise ValueError("X e y devono avere la stessa lunghezza.")

        splits = []
        N = len(X)
        test_size = int(N * self.test_size)


        for _ in range(self.num_splits):
            indices = np.arange(N) # Creazione di un array di indici per la suddivisione casuale
            np.random.shuffle(indices) # Mescola casualmente gli indici per garantire una suddivisione casuale
            
            test_indices = indices[:test_size] # Prime test_size istanze vengono assegnate al test set
            train_indices = indices[test_size:] # rimanenti assegnate al training set
            
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            
            splits.append((X_train, X_test, y_train, y_test))
        
        return splits
    
    def run(self, X, y, k=3):
        """
        Esegue il random subsampling e calcola l'accuratezza media del classificatore KNN.
        
        :param X: pandas DataFrame, features del dataset
        :param y: pandas Series, target del dataset
        :param k: numero di vicini per KNN
        
        :return (float): accuratezza media
        """
        accuracies = []
        splits = self.splitter(X, y)
        
        for X_train, X_test, y_train, y_test in splits:

            # inizializzazione e addestramento del modello KNN per ogni split
            knn = classificatore_KNN(k=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            # Calcolo dell'accuratezza basata sul numero di predizioni corrette
            correct = sum(y_pred == y_test)  # Conta il numero di predizioni corrette
            accuracies.append(correct / len(y_test))  # Calcola l'accuratezza come rapporto tra corrette e totale
        
        return np.mean(accuracies)
