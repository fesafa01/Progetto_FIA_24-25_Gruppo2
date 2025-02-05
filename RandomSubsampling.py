import numpy as np
from classificatore_KNN import classificatore_KNN
from metrics_calculator import metrics_calculator

"""
Questo modulo implementa il metodo di Random Subsampling per la valutazione di un modello di classificazione.

Il Random Subsampling consiste nel suddividere ripetutamente un dataset in training e test set in modo casuale,
eseguendo un numero predefinito di iterazioni. 
In ogni iterazione, il modello viene addestrato e testato,
e la prestazione viene misurata per ottenere una valutazione complessiva (data dalla media delle prestazioni singole).
"""

class RandomSubsampling:
    
    def __init__(self, test_size=0.3, num_splits=10):
        """
        Metodo costruttore per il random subsampling.
        test_size: Può essere:
            - Un valore in [0,1), interpretato come frazione (es. 0.3 = 30%)
            - Un valore in (1,100], interpretato come percentuale (es. 30 = 0.3)
            - Una stringa che termina con '%' (es. "30%")
        num_splits: numero di iterazioni per la suddivisione casuale.
        """

        # Se test_size è stringa, rimuoviamo il % e convertiamo a float
        if isinstance(test_size, str):
            # Rimuove eventuali spazi
            test_size = test_size.strip()
            
            # Se termina con "%"
            if test_size.endswith('%'):
                # rimuove il simbolo '%'
                test_size = test_size[:-1]
                # convertiamo a float
                test_size = float(test_size)
                # trasformiamolo in un valore percentuale
                test_size /= 100.0
            else:
                # Se l'utente ha scritto "0.3" come una stringa lo convertiamo in float
                test_size = float(test_size)

        # Se invece test_size è un numero, lo dividiamo per 100 per ottenere il valore percentuale
        if test_size > 1:
            test_size /= 100.0

        # infine verifichiamo che, come vuole il metodo successivo della classe, test_size sia tra 0 e 1
        if not 0 < test_size < 1:
            raise ValueError("test_size deve essere compreso tra 0 e 1, o tra 1 e 100 come percentuale, o stringa con '%'.")
    
        
        # Controllo della validità di num_splits, per superare rispwettivo test
        if not isinstance(num_splits, int) or num_splits <= 0:
            raise ValueError("num_splits deve essere un intero positivo maggiore di zero.")
        
        self.test_size = test_size
        self.num_splits = num_splits

    def splitter(self, X, y):
        """
        Divide il dataset in training e test set num_splits volte in modo casuale.
        
        :param X: pandas DataFrame, features del dataset
        :param y: pandas Series, target del dataset
        
        :return: lista di tuple (X_train, X_test, y_train, y_test) per ogni split
        """

        # Controllo che X e y non siano vuoti
        if X.empty or y.empty:
            raise ValueError("X e y non possono essere vuoti.")

        if len(X) != len(y):
            raise ValueError("X e y devono avere la stessa lunghezza.")

        splits = []
        N = len(X)
        test_size = int(N * self.test_size)


        for _ in range(self.num_splits):
            indices = np.arange(N) # Creazione di un array di indici per la suddivisione casuale
            np.random.shuffle(indices) # Mescola casualmente gli indici per garantire una suddivisione casuale
            
            test_indices = indices[:test_size] # Prime test_size istanze vengono assegnate al test set
            train_indices = indices[test_size:] # rimanenti assegnate al training set
            
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            
            splits.append((X_train, X_test, y_train, y_test))
        
        return splits
    
    def run(self, X, y, k=3):
        """
        Esegue il random subsampling e calcola l'accuratezza media del classificatore KNN.
        
        :param X: pandas DataFrame, features del dataset
        :param y: pandas Series, target del dataset
        :param k: numero di vicini per KNN
        
        :return: 
        - actual_value: lista contenente i valori reali delle classi nei vari test set.
        - predicted_value: lista contenente i valori predetti dal modello per i vari test set.
        """
    
        # inizializziamo due array vuoti che conterrano i valori predetti dal modello e quelli reali
        actual_value=[]
        predicted_value=[]
        predicted_score=[]
        splits = self.splitter(X, y)
        
        for X_train, X_test, y_train, y_test in splits:

            # inizializzazione e addestramento del modello KNN per ogni split
            knn = classificatore_KNN(k=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            y_score = knn.predict_proba(X_test)

            # Calcolo dell'accuratezza basata sul numero di predizioni corrette
            #correct = sum(y_pred == y_test)  # Conta il numero di predizioni corrette
            #accuracies.append(correct / len(y_test))  # Calcola l'accuratezza come rapporto tra corrette e totale

            # accumulo i valori all'interno dei due array
            actual_value.append(y_test)
            predicted_value.append(y_pred)
            predicted_score.append(y_score)
        
        return actual_value, predicted_value, predicted_score

    def evaluate(self,X,y,k=3):
        """
        Metodo per valutare le prestazioni del modello KNN utilizzando il metodo Random Subsampling.

        :param X: pandas DataFrame, rappresenta le feature del dataset.
        :param y: pandas Series, rappresenta le etichette di classe del dataset.
        :param k: int, numero di vicini da considerare per il classificatore KNN (default = 3).

        :return: dict, dizionario contenente le metriche di valutazione del modello.
        """
        # Eseguiamo il metodo run per ottenere i valori reali (actual) e predetti (predicted)
        actual_value, predicted_value, predicted_score = self.run(X,y,k)
        
        # Convertiamo le liste di pandas Series in array numpy monodimensionali
        actual_value = np.concatenate(actual_value).ravel()
        predicted_value = np.concatenate(predicted_value).ravel()
        predicted_score = np.concatenate(predicted_score).ravel()
        
        return actual_value, predicted_value, predicted_score
