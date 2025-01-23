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
        Funzione per lo splitting dei dati, riceve in ingresso features (X) e label (y) ed effettua lo splitting
        in training e test set
        :param X: pandas DataFrame, features del dataset
        :param y: pandas Series, label del dataset

        :return: X_train: pandas DataFrame, features del training set
        :return: X_test: pandas DataFrame, features del test set
        :return: y_train: pandas Series, label del training set
        :return: y_test: pandas Series, label del test set
        '''    
            
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
