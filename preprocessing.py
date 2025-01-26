# In questa classe sono riportate le funzioni per la pulizia e la normalizzazione dei dati, 
# la suddivisione in features e target e la rimozione di colonne non necessarie del dataset.


import numpy as np
import pandas as pd
import matplotlib as plot
import difflib

class Preprocessing:

    def __init__(self, dataset):
        self.dataset = dataset

    def filter_and_reorder_columns(self, dataset, threshold=0.7):
        """
        Mantiene nel dataset solo le colonne specificate, gestendo variazioni nei nomi e riordinandole nella sequenza
        in cui queste sono state fornite nella specifica del progetto.

        :param dataset: pandas DataFrame, il dataset originale
        :param threshold: float, soglia di similarità tra 0 e 1 (default 0.7)

        :return: pandas DataFrame, dataset con le colonne corrispondenti e nell'ordine specificato
        """
        columns_to_keep = [
            "Sample code number",
            "Clump Thickness",
            "Uniformity of Cell Size",
            "Uniformity of Cell Shape",
            "Marginal Adhesion",
            "Single Epithelial Cell Size",
            "Bare Nuclei",
            "Bland Chromatin",
            "Normal Nucleoli",
            "Mitoses",
            "Class"
        ]

        matched_columns = {}
        for col in columns_to_keep:
            match = difflib.get_close_matches(col, dataset.columns, n=1, cutoff=threshold)
            if match:
                matched_columns[col] = match[0]  # Mappa il nome corretto con quello effettivo nel dataset

        # Verifica se tutte le colonne richieste sono state trovate
        missing_columns = set(columns_to_keep) - set(matched_columns.keys())
        if missing_columns:
            print(f"Attenzione: le seguenti colonne non sono state trovate con la soglia impostata: {missing_columns}")

        # Seleziona le colonne trovate e le rinomina con i nomi desiderati
        dataset = dataset[list(matched_columns.values())].rename(columns={v: k for k, v in matched_columns.items()})

        # Riordina le colonne secondo l'ordine specificato in columns_to_keep
        dataset = dataset[columns_to_keep]

        return dataset

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
        Interpolazione dei valori NaN rimanenti con una media dei valori adiacenti.

        :param method: metodo di interpolazione (es. 'linear', 'polynomial', ecc.)

        :return: dataset con valori NaN interpolati
        '''
        # Converte le colonne a tipi numerici dove possibile, evitando errori
        self.dataset = self.dataset.infer_objects(copy=False)

        # Seleziona solo colonne numeriche per l'interpolazione
        numeric_cols = self.dataset.select_dtypes(include=['number'])

        # Applica l'interpolazione solo sulle colonne numeriche
        self.dataset[numeric_cols.columns] = numeric_cols.interpolate(method=method, axis=0)

        return self.dataset

    
    def normalize_data(self, dataset):
        '''
        Normalizzazione Min-Max per ogni colonna numerica (dalla seconda colonna in poi).
        
        La formula è: (X - X.min()) / (X.max() - X.min())

        :param dataset: pandas DataFrame
        :return: pandas DataFrame normalizzato
        '''
        # Converte i tipi oggetto in tipi appropriati (evita l'errore di interpolazione)
        dataset = dataset.infer_objects(copy=False)

        # Seleziona solo colonne numeriche dalla seconda colonna in poi
        numeric_cols = dataset.iloc[:, 1:].select_dtypes(include=['number'])

        # Applica la normalizzazione Min-Max solo alle colonne numeriche
        normalized_columns = numeric_cols.apply(lambda col: (col - col.min()) / (col.max() - col.min()))

        # Mantieni la prima colonna e unisci le colonne normalizzate
        self.dataset = pd.concat([dataset.iloc[:, [0]], normalized_columns], axis=1)

        return self.dataset


    
    def split(self, target):
        '''
        Suddivide il dataset in features (X) e target (y), escludendo la prima colonna.

        :param target: nome della colonna target
        :return: X: features (senza la colonna target), y: target
        '''
        # Mantieni la prima colonna intatta
        first_column = self.dataset.iloc[:, 0]

        # Rimuovi la prima colonna dal dataset temporaneo
        dataset_no_first_col = self.dataset.iloc[:, 1:]

        # Separazione di features (X) e target (y)
        X = dataset_no_first_col.drop(columns=[target])
        y = dataset_no_first_col[target]

        # Ricombina la prima colonna se necessario (opzionale, per riferimento futuro)
        self.dataset = pd.concat([first_column, X, y], axis=1)

        return X, y
