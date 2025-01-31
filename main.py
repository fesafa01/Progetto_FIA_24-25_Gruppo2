import numpy as np
import pandas as pd
import matplotlib as plot
from LogParser import LogParserFactory
from preprocessing import Preprocessing
from model_selection import ModelSelection as ms
import random
from classificatore_KNN import classificatore_KNN 
from Holdout import Holdout
from leave_one_out import leave_one_out
from RandomSubsampling import RandomSubsampling
from metrics_calculator import metrics_calculator

# Definiamo il nome del file su cui lavorare
filename = "version_1.csv"

# Usiamo la factory per creare il parser adatto al file
parser = LogParserFactory().create(filename)

# Carichiamo il dataset
dataset = parser.parse(filename)        

#Puliamo e riordiniamo il dataset secondo le specifiche della traccia
dataset = Preprocessing(dataset).filter_and_reorder_columns(dataset, 0.4)

#Pulisco il dataset dalle righe che contengono almeno 3 NaN
dataset = Preprocessing(dataset).drop_nan(8)

#Elimino le righe che non hanno valore nella class label
dataset = Preprocessing(dataset).drop_nan_target("Class")

#Riempio i valori NaN rimanenti con una media dei valori adiacenti
dataset = Preprocessing(dataset).interpolate("linear")

#Suddivido in features (X) e target (y)
X, y = Preprocessing(dataset).split("Class")

# Normalizzo le features
X = Preprocessing(dataset).normalize_data(X)

print(dataset)
print(X)
print(y)

#Scelgo il metodo di divisione del dataset in train e test set
choice = ms.model_selection()

if choice == "1":
    # Metodo Holdout
    test_size = float(input("Inserisci il valore percentuale che rappresenta la dimensione del test set rispetto all'intero set di dati: "))
    k = int(input("Inserisci il valore di k pari al numero di vicini da considerare: "))
    holdout = Holdout(test_size)
    metriche = holdout.evaluate(X,y,k)  #Calcoliamo tutte le metriche
    metriche_da_stampare = metrics_calculator() # Inizializzazione
    metriche_da_stampare.stampa_metriche(metriche) # Chiamata al metodo che stampa solo le metriche desiderate
    

elif choice == "2":
        # Metodo Leave One Out
        k = int(input("Inserisci il valore di k pari al numero di vicini da considerare: "))
        # Numero di esperimenti richiesto dall'utente:
        K = int(input(f"Inserisci il numero di esperimenti (intero) tra 1 e {len(X)}: "))
        Leave_one_out = leave_one_out(K, X) #Inizializzazione
        metriche = Leave_one_out.evaluate(X,y,k)
        metriche_da_stampare = metrics_calculator() # Inizializzazione
        metriche_da_stampare.stampa_metriche(metriche) # Chiamata al metodo che stampa solo le metriche desiderate
        

elif choice == "3":
        # Metodo Random Subsampling
        test_size = float(input("Inserisci il valore percentuale che rappresenta la dimensione del test set rispetto all'intero set di dati: "))
        k = int(input("Inserisci il valore di k pari al numero di vicini da considerare: "))
        num_splits = int(input("Inserisci il numero di splits da realizzare nel metodo: "))
        random_subsampling = RandomSubsampling(test_size, num_splits) #Inizializzazione
        metriche = random_subsampling.evaluate(X,y,k)
        
        metriche_da_stampare = metrics_calculator() # Inizializzazione
        metriche_da_stampare.stampa_metriche(metriche) # Chiamata al metodo che stampa solo le metriche desiderate
else:
        print("Scelta non valida. Riprova.")

