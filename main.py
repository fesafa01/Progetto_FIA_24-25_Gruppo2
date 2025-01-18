import numpy as np
import pandas as pd
import matplotlib as plot

#Carichiamo il dataset assegnato
dataset = pd.read_csv("version_1.csv")

#Pulisco il dataset dalle righe che contengono almeno 3 NaN
dataset = dataset.dropna(thresh=10)

#Elimino le righe che non hanno la classtype specificata
dataset = dataset.dropna(subset=["classtype_v1"])

#Riempio i valori NaN rimanenti con una media dei valori adiacenti
dataset = dataset.interpolate(method='linear', axis=0)

# Elimino la prima colonna relativa al numero di osservazioni
dataset = dataset.iloc[:, 1:]

#Suddivido in features (X) e target (y)
X = dataset.drop(columns=["classtype_v1"])
y = dataset["classtype_v1"]