# Funzione che implementa un classificatore K-Nearest Neighbors (KNN) per la classificazione di un dataset.
# Il classificatore KNN è un algoritmo di apprendimento supervisionato che classifica un punto di dati in base ai suoi vicini più vicini.

import numpy as np
import random

class classificatore_KNN:
    
    def __init__(self, X, y, k=3, fun_distanza=None):
        self.k = k  # Numero di vicini considerati
        self.X_train = X  # Carica dati di training
        self.y_train = y  # Carica labels di training
        # Imposta la funzione di distanza, default è la distanza euclidea
        if fun_distanza is not None:
            self.fun_distanza = fun_distanza
        else: self.fun_distanza = self.distanza_euclidea

    # Funzione per calcolare la distanza euclidea tra due punti x1 e x2 
    def distanza_euclidea(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))


    def predict(self, X_test):
        # Prevedere le labels per ciascun punto nel set di tst
        predizioni = []

        for x in X_test:
            # Calcolare la distanza euclidea da tutti i punti di training
            distanze = []
            for x_train in self.X_train:
                distanza = self.fun_distanza(x_train, x)
                distanze.append(distanza)
            
            # Ottenere gli indici dei k punti più vicini
            k_indici = np.argsort(distanze)[:self.k]
            

            # Raccogliere le labe,s dei k vicini
            k_nearest_labels = []
            for ind in k_indici:
                k_nearest_labels.append(self.y_train[ind])
            
            # Determinare la label più comune tra i vicini
            predizioni.append(self.majority_vote(k_nearest_labels))
        
        return predizioni

    def majority_vote(self, labels):
        # Contare le frequenze delle etichette
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

