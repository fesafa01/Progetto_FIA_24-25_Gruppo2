import numpy as np
import pandas as pd
import matplotlib as plot
from preprocessing import Preprocessing
from model_selection import ModelSelection as ms
import random
from classificatore_KNN import classificatore_KNN 
from Holdout import Holdout

#Carichiamo il dataset assegnato
dataset = pd.read_csv("version_1.csv")

#Puliamo e riordiniamo il dataset secondo le specifiche della traccia
dataset = Preprocessing(dataset).filter_and_reorder_columns(dataset, 0.4)

#Pulisco il dataset dalle righe che contengono almeno 3 NaN
dataset = Preprocessing(dataset).drop_nan(8)

#Elimino le righe che non hanno valore nella class label
dataset = Preprocessing(dataset).drop_nan_target("Class")

#Riempio i valori NaN rimanenti con una media dei valori adiacenti
dataset = Preprocessing(dataset).interpolate("linear")

# Normalizzo i dati
dataset = Preprocessing(dataset).normalize_data(dataset)

print(dataset)

#Suddivido in features (X) e target (y)
X, y = Preprocessing(dataset).split("Class")

#Scelgo il metodo di divisione del dataset in train e test set
choice = ms.model_selection()

if choice == "1":
    # Metodo Holdout
    test_size = float(input("Inserisci il valore percentuale che rappresenta la dimensione del test set rispetto all'intero set di dati: "))
    holdout = Holdout(test_size)
    X_train, X_test, y_train, y_test = holdout.splitter(X, y)
elif choice == "2":
        # Metodo Leave One Out
        pass
elif choice == "3":
        # Metodo 3
        pass
else:
        print("Scelta non valida. Riprova.")
