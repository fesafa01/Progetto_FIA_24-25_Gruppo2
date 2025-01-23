# In questa classe sono riportate le funzioni per la pulizia e la normalizzazione dei dati, 
# la suddivisione in features e target e la rimozione di colonne non necessarie del dataset.


import numpy as np
import pandas as pd
import matplotlib as plot

class Preprocessing:

    def __init__(self, dataset):
        self.dataset = dataset

    def drop_nan(self, threshold):
        '''
        Elimina le righe che contengono almeno (#colonne - threshhold) NaN
        quindi in pratica tengo le righe che hanno almeno threshold valori non NaN

        :param threshold: soglia di valori non NaN

        :return: dataset senza righe con almeno threshold valori NaN
        '''
        self.dataset = self.dataset.dropna(thresh=threshold)
        return self.dataset
 
    def drop_nan_target(self, target):
        '''
        Elimina le righe che non hanno valore nella colonna target

        :param target: nome della colonna target

        :return: dataset senza righe con valori NaN nella colonna target
        '''
        self.dataset = self.dataset.dropna(subset=[target])
        return self.dataset 

    def interpolate(self, method):
        '''
        Interpolazione dei valori NaN rimanenti con una media dei valori adiacenti

        :param method: metodo di interpolazione

        :return: dataset con valori NaN interpolati
        '''
        self.dataset = self.dataset.interpolate(method=method, axis=0)
        return self.dataset
    
    def normalize_data(self, dataset):
        '''
        Normalizzazione Min-Max per ogni colonna.
        La formula è (X - X.min) / (X.max - X.min), apply consente di applicare la funzione a tutte le colonne del dataset.
        .apply() applica la funzione lambda a tutte le colonne del dataset  e restituisce il dataset normalizzato
        '''
        self.dataset = dataset.apply(lambda col: (col - col.min()) / (col.max() - col.min()))
        return self.dataset
    
    def split(self, target):
        '''
        Suddivide il dataset in features (X) e target (y)
        :param target: nome della colonna target
        :return: X: features, y: target
        '''
        X = self.dataset.drop(columns=[target])
        y = self.dataset[target]
        return X, y     