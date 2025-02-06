import numpy as np
import random
import pandas as pd
from classificatore.Distance import DistanceFactory

class classificatore_KNN:
    """classificatore basato sull'algoritmo k-Nearest Neighbors
    
    Parametri:
        choice_distance: str, tipo di distanza da utilizzare.
        k (int, opzionale): Il numero di vicini da considerare (Default è 3)
        Se fun_distanza è None, utilizza la distanza euclidea come default.

    Attributi:
        X_train (pd.DataFrame): dati di training.
        y_train (pd.Series): labels di training.
        k (int):  numero di vicini da considerare.
        fun_distanza (callable): Il tipo di distanza da calcolare tra due punti.
    """
    

    def __init__(self, choice_distance=1, k=3):
        """
        Inizializza il classificatore con un numero specificato di vicini (k) e un tipo di distanza.
        
        Parametri:
        - choice_distance: str, tipo di distanza da utilizzare (euclidea come default).
        - k (int): Numero di vicini considerati per il calcolo della predizione. Default è 3.
        
        Output:
        -Nessun valore di ritorno (None).
        """
        distance = DistanceFactory.create(self,choice_distance)
        
        self.k = k  # Numero di vicini considerati

        # Imposta il tipo di distanza, default è la distanza euclidea
        self.fun_distanza = distance

        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        Addestra il classificatore KNN con i dati di training forniti

        Parametri:
        -X (pd.DataFrame): DataFrame contenente i dati di training
        -y (pd.Series): Series contenente le labels di training
        
        Output:
        -Nessun valore di ritorno (None).
        """
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("I dati inseriti non sono DataFrame!")
        if not isinstance(y, pd.Series):
            raise ValueError("Le labels inserite non sono Series!")
        self.X_train = X
        self.y_train = y


    def predict(self, X_test):
        """
        Prevede le etichette per il dataset di test
        
        Parametri:
        - X_test (DataFrame): Features del dataset di test.
        
        Output:
        - Series: Etichette predette per il dataset di test.
        """
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Addestrare con il metodo 'fit' prima di prevedere.")

        # Calcola tutte le distanze tra i punti di test e quelli di training
        distanze = self.fun_distanza.calcola(X_test.values, self.X_train.values)
        
        # Ottenere gli indici dei k punti più vicini
        k_indici = np.argsort(distanze, axis=1)[:, :self.k]
        
        # Raccogliere le labe,s dei k vicini
        k_nearest_labels = []
        for ind in k_indici:
            k_nearest_labels.append(self.y_train.iloc[ind])
            
        
        # Determinare la label più comune tra i vicini
        predizioni = [self.majority_vote(labels) for labels in k_nearest_labels]

        # Restituisce una Series con le predizioni
        return pd.Series(predizioni, index=X_test.index)
    
    
    ########################
    # NUOVO METODO PER SCORE
    ########################

    def predict_proba(self, X_test):
        """
        Calcola le probabilità che ogni istanza di X_test appartenga alla classe positiva.
        
        Args:
            X_test (pd.DataFrame): Il set di dati di test.
        
        Returns:
            pd.Series: Serie di probabilità per la classe positiva.
        """
        positive_class = 2 # Definizione della classe considerata positiva
        
        # Calcolo delle distanze tra ogni punto in X_test e tutti i punti in X_train
        distanze = self.fun_distanza.calcola(X_test.values, self.X_train.values)
        
        # Trova gli indici dei k vicini più prossimi ordinando per distanza crescente
        k_indici = np.argsort(distanze, axis=1)[:, :self.k]
        
        # Recupera le etichette dei k vicini più prossimi
        k_nearest_labels = [self.y_train.iloc[ind] for ind in k_indici]
        
        probabilities = []
        for labels in k_nearest_labels:
            # Calcola la probabilità come la media delle etichette uguali alla classe positiva
            prob = np.mean(labels == positive_class)
            probabilities.append(prob)
        
        # Converte le probabilità in una Serie pandas con gli stessi indici di X_test
        predicted_scores = pd.Series(probabilities, index=X_test.index)
        
        return predicted_scores

    def majority_vote(self, labels):
        """
        Determina l'etichetta più comune tra quelle in input
        
        Parametri:
        - labels (Series): etichette ricevute dai k vicini.
        
        Output:
        - string: L'etichetta che ha ricevuto più voti.
        """
        
        label_count = {}
        for label in labels:
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1
        
        # Determinare le label più frequenti
        max_votes = max(label_count.values())
        label_vincitrici = [label for label, count in label_count.items() if count == max_votes]
        
        # Selezionare casualmente in caso di pareggio
        return random.choice(label_vincitrici) if len(label_vincitrici) > 1 else label_vincitrici[0]

    import numpy as np