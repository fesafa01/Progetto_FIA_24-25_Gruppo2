import pandas as pd
import numpy as np

#Implementazione del metodo di validazione Holdout, che divide il dataset in due parti: training e test set
# in particolare, il test set è una porzione del dataset che non verrà utilizzata per l'addestramento del modello
# ma solo per la sua valutazione

class Holdout:
        
    #Definiamo la dimensione dello splitting dei dati
    def __init__(self, test_size):
        if not 0 < test_size < 1:
            raise ValueError("test_size deve essere compreso tra 0 e 1.")
        self.test_size = test_size

    #Funzione per lo splitting dei dati, riceve in ingresso features (X) e label (y) ed effettua lo splitting
    def splitter(self, X, y):
        # Calcola il numero di campioni per il test set
        test_size = int(len(X) * self.test_size)
        
        # Divido X in train e test
        test_indices = X[:test_size]
        train_indices = X[test_size:]

        #Controllo se X e y sono vuoti
        if X.empty or y.empty:
            raise ValueError("X e y non possono essere vuoti.")

        
        # Suddivisione del dataset
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
     
        return X_train, X_test, y_train, y_test
