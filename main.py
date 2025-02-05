from LogParser import LogParserFactory
from preprocessing import Preprocessing
from model_selection import ModelSelection as ms
from metrics_calculator import metrics_calculator
from ModelEvaluation import ModelEvaluationFactory
import os
from metrics_visualizer import metrics_visualizer

# Definiamo il valore di default per il file
DEFAULT_FILENAME = "version_1.csv"

while True:
    # Chiediamo all'utente il file da usare (con possibilità di lasciare il default)
    user_input = input(f"Inserisci il percorso del file (premi Invio per usare il default: {DEFAULT_FILENAME}): ").strip()

    # Se l'utente non inserisce nulla, usa il file di default
    filename = user_input if user_input else DEFAULT_FILENAME

    # Controlliamo se il file esiste per evitare errori
    if os.path.isfile(filename):
        break  # Esce dal ciclo se il file esiste
    else:
        print(f"Errore: Il file '{filename}' non esiste. Riprova.")

print(f"File selezionato: {filename}")


# Nel seguente codice commentato, proponiamo una soluzione per l'inserimento del percorso e del file
# contenente il dataset attraverso una finestra in cui l'utente può selezionare il file.
# Abbiamo preferito la soluzione con l'inserimento manuale del path ma lasciamo qui la possibilità di modificare questa preferenza.
"""
import tkinter as tk
from tkinter import filedialog

# Apriamo una finestra di selezione file
root = tk.Tk()
root.withdraw()  # Nasconde la finestra principale
filename = filedialog.askopenfilename(title="Seleziona il file CSV")

# Se l'utente non sceglie nulla, usa il file di default
if not filename:
    filename = DEFAULT_FILENAME

# Controlliamo se il file esiste
if not os.path.isfile(filename):
    print(f"Errore: Il file '{filename}' non esiste. Assicurati che il percorso sia corretto.")
    exit(1)
"""

# Usiamo la factory per creare il parser adatto al file
parser = LogParserFactory().create(filename)

# Carichiamo il dataset
dataset = parser.parse(filename)        

#Puliamo e riordiniamo il dataset secondo le specifiche della traccia
dataset = Preprocessing.filter_and_reorder_columns(dataset, 0.4)

#Pulisco il dataset dalle righe che contengono almeno 3 NaN
dataset = Preprocessing.drop_nan(dataset, 8)

#Elimino le righe che non hanno valore nella class label
dataset = Preprocessing.drop_nan_target(dataset, "Class")

#Seleziono come sostituire i valori NaN nel dataset scegliendo tra media, mediana, moda e rimuovi
indicatore = str(input("Inserisci il metodo di sostituzione dei valori NaN (media, mediana, moda, rimuovi): "))
dataset = Preprocessing.nan_handler(dataset, indicatore)

#Suddivido in features (X) e target (y)
X, y = Preprocessing.split(dataset, "Class")

# Seleziono il metodo di feature scaling delle features del dataset
method = int(input("Scegli il metodo di feature scaling: \n1 - Standardizzazione \n2 - Normalizzazione\n"))
X = Preprocessing.feature_scaling(X, method)

'''
print(dataset)
print(X)
print(y)
'''

#Scelgo il metodo di divisione del dataset in train e test set
choice = ms.model_selection()
k = int(input("Inserisci il valore di k pari al numero di vicini da considerare: "))

if choice == "1":
    test_size = input("Inserisci il valore percentuale per il test set: ")
    strategy = ModelEvaluationFactory.get_validation_strategy(choice, test_size=test_size)

elif choice == "2":
        K = int(input(f"Inserisci il numero di esperimenti (intero) tra 1 e {len(X)}: "))
        strategy = ModelEvaluationFactory.get_validation_strategy(choice, K=K)

elif choice == "3":
    test_size = input("Inserisci il valore percentuale per il test set: ")
    num_splits = int(input("Inserisci il numero di splits: "))
    strategy = ModelEvaluationFactory.get_validation_strategy(choice, test_size=test_size, num_splits=num_splits)
else:
    raise ValueError("Scelta non valida.")

# Ora possiamo usare strategy
actual_value, predicted_value = strategy.evaluate(X, y, k) # Calcolo delle metriche

#Inizializziamo oggetto per visualizzare metriche e salvarle
MetricsVisualizer = metrics_visualizer(actual_value, predicted_value)

# Visualizziamo le metrche del modello
MetricsVisualizer.visualizza_metriche()

#Salviamo le metriche in relativo file excel (Dare in input il nome del file, default:"model_performance.xlsx")
MetricsVisualizer.save()


