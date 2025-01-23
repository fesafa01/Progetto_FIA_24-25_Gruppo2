import numpy as np
import pandas as pd
import matplotlib as plot
from preprocessing import Preprocessing
from model_selection import ModelSelection as ms
import random
from classificatore_KNN import classificatore_KNN 
from holdout import Holdout

#Carichiamo il dataset assegnato
dataset = pd.read_csv("version_1.csv")

#Pulisco il dataset dalle righe che contengono almeno 3 NaN
dataset = Preprocessing(dataset).drop_nan(10)

#Elimino le righe che non hanno la classtype specificata
dataset = Preprocessing(dataset).drop_nan_target("classtype_v1")

#Riempio i valori NaN rimanenti con una media dei valori adiacenti
dataset = Preprocessing(dataset).interpolate("linear")

# Elimino la prima colonna relativa al numero di osservazioni
dataset =  Preprocessing(dataset).drop_column("Unnamed: 0")

# Normalizzo i dati
dataset = Preprocessing(dataset).normalize_data(dataset)

#Suddivido in features (X) e target (y)
X, y = Preprocessing(dataset).split("classtype_v1")

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
