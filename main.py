from LogParser import LogParserFactory
from preprocessing import Preprocessing
from model_selection import ModelSelection as ms
from metrics_calculator import metrics_calculator
from ModelEvaluation import ModelEvaluationFactory

# Definiamo il nome del file su cui lavorare
filename = "Progetto_FIA_24-25_Gruppo2/version_1.csv"

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

ModelEvaluation = ModelEvaluationFactory().get_validation_strategy(choice, X)
k = int(input("Inserisci il valore di k pari al numero di vicini da considerare: "))
metriche = ModelEvaluation.evaluate(X, y, k)
metriche_da_stampare = metrics_calculator() # Inizializzazione
metriche_da_stampare.stampa_metriche(metriche) # Chiamata al metodo che stampa solo le metriche desiderate

'''
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

'''