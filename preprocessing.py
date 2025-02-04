# In questa classe sono riportate le funzioni per la pulizia e la normalizzazione dei dati, 
# la suddivisione in features e target e la rimozione di colonne non necessarie del dataset.

import pandas as pd
from rapidfuzz import process, fuzz

class Preprocessing:

    @staticmethod
    def normalize_column_name(name):
        """ Normalizza il nome delle colonne rimuovendo underscore, spazi extra e mettendo tutto in minuscolo. """
        return name.lower().replace("_", " ").replace("-", " ")

    @staticmethod
    def filter_and_reorder_columns(dataset, threshold=70):
        """
        Mantiene nel dataset solo le colonne specificate, gestendo variazioni nei nomi e riordinandole nella sequenza
        in cui queste sono state fornite nella specifica del progetto.

        :param dataset: pandas DataFrame, il dataset originale
        :param threshold: int, soglia di similarità tra 0 e 100 (default 70)

        :return: pandas DataFrame, dataset con le colonne corrispondenti e nell'ordine specificato
        """
        columns_to_keep = [
            "Sample code number",
            "Clump Thickness",
            "Uniformity of Cell Shape",
            "Uniformity of Cell Size",
            "Marginal Adhesion",
            "Single Epithelial Cell Size",
            "Bare Nuclei",
            "Bland Chromatin",
            "Normal Nucleoli",
            "Mitoses",
            "Class"
        ]

        # Normalizza solo i nomi delle colonne, non il dataset intero
        dataset.columns = [Preprocessing.normalize_column_name(col) for col in dataset.columns]
        normalized_columns_to_keep = {col: Preprocessing.normalize_column_name(col) for col in columns_to_keep}

        matched_columns = {}

        for col in columns_to_keep:
            # Filtra le colonne già assegnate per evitare duplicati
            available_columns = list(set(dataset.columns) - set(matched_columns.values()))

            # Trova la colonna più simile disponibile
            match, score, _ = process.extractOne(
                normalized_columns_to_keep[col], 
                available_columns,
                scorer=fuzz.WRatio
            )

            if match and score >= threshold:
                matched_columns[col] = match
                print(f"Trovata corrispondenza: {col} → {match} (score: {score})")  # Debug
            else:
                print(f"Nessuna corrispondenza valida per: {col} (score massimo: {score})")  # Debug

        # Gestisce le colonne mancanti senza generare errori
        missing_columns = set(columns_to_keep) - set(matched_columns.keys())
        if missing_columns:
            print(f"Attenzione: le seguenti colonne non sono state trovate: {missing_columns}")

        # Creiamo un DataFrame con colonne trovate, riempiendo quelle mancanti con NaN
        dataset = dataset[list(matched_columns.values())].rename(columns={v: k for k, v in matched_columns.items()})
        
        for missing_col in missing_columns:
            dataset[missing_col] = None  # Inserisce NaN per le colonne mancanti

        dataset = dataset[columns_to_keep]  # Riordino

        return dataset

    @staticmethod
    def drop_nan(dataset, threshold):
        '''
        Elimina le righe che contengono almeno (#colonne - threshhold) NaN
        quindi in pratica tengo le righe che hanno almeno threshold valori non NaN

        :param dataset: dataset da pulire
        :param threshold: soglia di valori non NaN

        :return: dataset senza righe con almeno threshold valori NaN
        '''
        dataset = dataset.dropna(thresh=threshold)
        return dataset
 
    @staticmethod   
    def drop_nan_target(dataset, target):
        '''
        Elimina le righe che non hanno valore nella colonna target

        :param dataset: dataset da pulire
        :param target: nome della colonna target

        :return: dataset senza righe con valori NaN nella colonna target
        '''
        dataset = dataset.dropna(subset=[target])
        return dataset 

    @staticmethod
    def nan_handler(dataset, method):
        '''
        Sostituzione dei valori NaN con media, mediana, moda della rispettiva colonna o rimozione delle righe con valori NaN.

        :param dataset: dataset da pulire
        :param method: metodo di sostituzione ('media', 'mediana', 'moda', 'rimuovi')
        
        :return: dataset con valori NaN sostituiti o eliminati
        '''
        # Converte tutte le colonne in numerico dove possibile, impostando NaN per i valori non convertibili
        dataset = dataset.apply(pd.to_numeric, errors='coerce')

        if method == 'media':
            dataset = dataset.fillna(dataset.mean())
        elif method == 'mediana':
            dataset = dataset.fillna(dataset.median())
        elif method == 'moda':
            dataset = dataset.apply(lambda col: col.fillna(col.mode()[0]) if not col.mode().empty else col)
        elif method == 'rimuovi':
            dataset = dataset.dropna()
        else:
            raise ValueError("Metodo non valido! Usa 'mean', 'median', 'mode' o 'remove'.")
        
        return dataset

    @staticmethod
    def feature_scaling(X, method=1):
        '''
        Funzione per scalare le features del modello.
        
        Opzioni disponibili:
        1 - Normalizzazione Min-Max: (X - X.min()) / (X.max() - X.min())
        2 - Standardizzazione Z-score: (X - X.mean()) / X.std()
        
        :param X: features del modello
        :param method: 1 per Normalizzazione, 2 per Standardizzazione. Se non specificato applica Normalizzazione.
        :return X: features scalate
        '''
        # Converte tutto a numerico e fornisce NaN per valori non convertibili
        X = X.apply(pd.to_numeric, errors='coerce')
        
        if method == 1:
            # Normalizzazione Min-Max
            X = (X - X.min()) / (X.max() - X.min())
        elif method == 2:
            # Standardizzazione
            X = (X - X.mean()) / X.std()
        else:
            raise ValueError("Metodo non valido! Usa 1 per Normalizzazione o 2 per Standardizzazione.")

        return X


    @staticmethod
    def split(dataset, target):
        '''
        Suddivide il dataset in features (X) e target (y), escludendo la prima colonna.

        :param dataset: dataset da suddividere
        :param target: nome della colonna target
        :return: X: features (senza la colonna target), y: target
        '''
        # Mantieni la prima colonna intatta
        first_column = dataset.iloc[:, 0]

        # Rimuovi la prima colonna dal dataset temporaneo
        dataset_no_first_col = dataset.iloc[:, 1:]

        # Separazione di features (X) e target (y)
        X = dataset_no_first_col.drop(columns=[target])
        y = dataset_no_first_col[target]

        return X, y
