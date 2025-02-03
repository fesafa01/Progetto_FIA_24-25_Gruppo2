from LogParser import LogParserFactory
from preprocessing import Preprocessing
from model_selection import ModelSelection as ms
from metrics_calculator import metrics_calculator
from ModelEvaluation import ModelEvaluationFactory

# Definiamo il nome del file su cui lavorare
filename = "version_1.csv"

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

#Riempio i valori NaN rimanenti con una media dei valori adiacenti
dataset = Preprocessing.interpolate(dataset, "linear")

#Suddivido in features (X) e target (y)
X, y = Preprocessing.split(dataset, "Class")

# Normalizzo le features
X = Preprocessing.normalize_data(X)

print(dataset)
print(X)
print(y)

#Scelgo il metodo di divisione del dataset in train e test set
choice = ms.model_selection()
k = int(input("Inserisci il valore di k pari al numero di vicini da considerare: "))

if choice == "1":
    test_size = float(input("Inserisci il valore percentuale per il test set: "))
    strategy = ModelEvaluationFactory.get_validation_strategy(choice, test_size=test_size)

elif choice == "2":
        K = int(input(f"Inserisci il numero di esperimenti (intero) tra 1 e {len(X)}: "))
        strategy = ModelEvaluationFactory.get_validation_strategy(choice, K=K)

elif choice == "3":
    test_size = float(input("Inserisci il valore percentuale per il test set: "))
    num_splits = int(input("Inserisci il numero di splits: "))
    strategy = ModelEvaluationFactory.get_validation_strategy(choice, test_size=test_size, num_splits=num_splits)
else:
    raise ValueError("Scelta non valida.")

# Ora possiamo usare strategy
metriche = strategy.evaluate(X, y, k) # Calcolo delle metriche
metriche_da_stampare = metrics_calculator() # Inizializzazione
metriche_da_stampare.stampa_metriche(metriche) # Chiamata al metodo che stampa solo le metriche desiderate
print("Metriche richieste:", metriche_da_stampare)

