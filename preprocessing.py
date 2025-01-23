# In questa classe sono riportate le funzioni per la pulizia e la normalizzazione dei dati, 
# la suddivisione in features e target e la rimozione di colonne non necessarie del dataset.


import numpy as np
import pandas as pd
import matplotlib as plot

class Preprocessing:

    def __init__(self, dataset):
        self.dataset = dataset

    #Pulisco il dataset dalle righe che contengono almeno n righe - threshhold NaN
    def drop_nan(self, threshold):
        self.dataset = self.dataset.dropna(thresh=threshold)
        return self.dataset

    #Elimino le righe che non hanno valore nella colonna target 
    def drop_nan_target(self, target):
        self.dataset = self.dataset.dropna(subset=target)
        return self.dataset 

    #Riempio i valori NaN rimanenti con una media dei valori adiacenti
    def interpolate(self, method):
        self.dataset = self.dataset.interpolate(method=method, axis=0)
        return self.dataset
    
    #Elimino la prima colonna relativa al numero di osservazioni
    def drop_column(self, column):  
        self.dataset = self.dataset.drop(columns=column)
        return self.dataset

    def normalize_data(self, dataset):
        # Normalizzazione Min-Max per ogni colonna. 
        # La formula è (X - X.min) / (X.max - X.min), apply consente di applicare la funzione a tutte le colonne del dataset
        self.dataset = dataset.apply(lambda col: (col - col.min()) / (col.max() - col.min()))
        return self.dataset
    
    #Suddivido in features (X) e target (y)
    def split(self, target):
        X = self.dataset.drop(columns=target)
        y = self.dataset[target]
        return X, y 
    