from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from classificatore_KNN import classificatore_KNN
from metrics_calculator import metrics_calculator

class ModelSelection(ABC):
    '''
    Classe astratta per la selezione del metodo di validazione del modello.
    '''
    def __init__(self, test_size=None, K=None, num_splits=None):
        """
        Costruttore della classe base per tutte le strategie di validazione.

        :param test_size: float, dimensione del test set (se applicabile).
        :param K: int, numero di iterazioni per Leave-One-Out (se applicabile).
        :param num_splits: int, numero di suddivisioni per Random Subsampling.
        """
        self.test_size = test_size
        self.K = K
        self.num_splits = num_splits

    @abstractmethod
    def splitter(self, X, y):
        """Metodo astratto per dividere il dataset in training e test set."""
        pass

    @abstractmethod
    def run(self, X, y, k=3):
        """Metodo astratto per eseguire la validazione e testare il modello."""
        pass

    @abstractmethod
    def evaluate(self, X, y, k=3):
        """Metodo astratto per calcolare le metriche di valutazione."""
        pass

class Holdout(ModelSelection):
        
    def __init__(self, test_size=0.3):
        '''
        Costruttore della classe Holdout
        :param test_size: float, valore percentuale che rappresenta la 
        dimensione del test set rispetto all'intero set di dati, di default è 0.3
        in output non ritorna nulla

        '''
        if not 0 < test_size < 1:
            raise ValueError("test_size deve essere compreso tra 0 e 1.")
        self.test_size = test_size

    
    def splitter(self, X, y):
        """
        Suddivide il dataset in training e test set in base alla proporzione specificata.

        Il metodo esegue un mescolamento casuale degli indici del dataset, quindi divide i dati in due insiemi:
        - Training set: usato per addestrare il modello
        - Test set: usato per valutare le prestazioni del modello

        :param X: pandas DataFrame, rappresenta le feature del dataset.
        :param y: pandas Series, rappresenta le etichette di classe del dataset.

        :return:
            - X_train: subset del dataset usato per l'addestramento.
            - y_train: etichette corrispondenti al training set.
            - X_test: subset del dataset usato per il test.
            - y_test: etichette corrispondenti al test set.
        """

        # Controllo se X e y sono vuoti
        if X.empty or y.empty:
            raise ValueError("X e y non possono essere vuoti.")

        # Controllo che X e y abbiano la stessa lunghezza
        if len(X) != len(y):
            raise ValueError("X e y devono avere la stessa lunghezza.")

        # Shuffle degli indici
        indices = np.arange(len(X))  # Genera un array di indici numerici
        np.random.shuffle(indices)

        # Calcola il numero di campioni per il test set
        test_size = int(len(X) * self.test_size)

        # Divisione degli indici numerici in train e test
        train_indices = indices[test_size:]  # Indici per il training set
        test_indices = indices[:test_size]   # Indici per il test set

        # Suddivisione del dataset usando .iloc con indici numerici validi
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        return X_train, y_train, X_test, y_test

    def run(self, X,y,k=3):
        """
        Addestra e testa un classificatore KNN utilizzando la suddivisione del dataset in training e test set.
        
        :param X: pandas DataFrame, rappresenta le feature del dataset.
        :param y: pandas Series, rappresenta le etichette di classe del dataset.
        :param k: int, numero di vicini da considerare per il classificatore KNN.

        :return:
            - actual_value: etichette reali del test set.
            - predicted_value: valori predetti dal modello KNN sul test set.
        """
        
        
        # chiamiamo il metodo precedente per ottenere X_train, y_train, X_test, y_test
        X_train, y_train, X_test, y_test = self.splitter(X,y)

        # si crea il KNN con il parametro k specifico per l'iterazione a cui ci troviamo
        knn = classificatore_KNN(k)
            
        # si allena il modello sul training set specifico per l'iterazione a cui ci troviamo
        knn.fit(X_train, y_train)
            
        # si fa una predizione su y
        y_pred = knn.predict(X_test) 

        # si salvano in due variabili il valore corretto e quello predetto dal modello
        actual_value = y_test
        predicted_value = y_pred

        return actual_value, predicted_value

    def evaluate(self, X,y,k=3):
        """
        Valuta le prestazioni del classificatore KNN utilizzando le metriche di validazione.

        :param X: pandas DataFrame, rappresenta le feature del dataset.
        :param y: pandas Series, rappresenta le etichette di classe del dataset.
        :param k: int, numero di vicini da considerare per il classificatore KNN.

        :return:
            - metriche: dizionario contenente le metriche di valutazione del modello
        """
        
        # Eseguiamo il metodo run per ottenere i valori reali (actual) e predetti (predicted)
        actual_value, predicted_value = self.run(X,y,k)
        # creiamo un'istanza della classe che calcola le metriche di valutazione del modello
        Calculator = metrics_calculator()
        # calcoliamo la matrice di confusione
        matrix = Calculator.confusion_matrix(predicted_value, actual_value)
        # calcoliamo le metriche
        metriche = Calculator.metrics_evalutation(matrix, predicted_value, actual_value)
    
        return metriche

#Implementazione del metodo di validazione Leave One Out, che divide per tante volte quanto è il numero di esperimenti richiesto, 
# il dataset in due parti: training set e test set.
# in questo metodo di validazione in ogni iterazione si sceglie casualmente un campione che costituirà il test set, 
# i restanti campioni costituiranno invece il training set. In questo modo avremo la massima percentuale di campioni 
# impiegata per addestrare l'algoritmo.
# Si calcolano inoltre metriche di validazione per il modello.

class LeaveOneOut(ModelSelection):
        
    def __init__(self, K: int, X):
        """
        Metodo costruttore del leave_one_out.
        K è il numero di esperimenti che si eseguirà, il suo valore è dato dall'utente
        Per il corretto funzionamento dei metodi successivi è necessario che K sia un intero.
        """
        if not isinstance(K, int):  # Controllo che K sia un intero
            raise ValueError("K deve essere un numero intero.")
        
        if not 1 <= K <= len(X):  # Controlla che K sia valido
            raise ValueError(f"Errore: il numero di esperimenti deve essere compreso tra 1 e {len(X)}.")
        
        self.K = K
        self.X = X 
    

    def splitter (self, X, y):
        '''
        Metodo che  divide il dataset in training set e test set K volte.
        parametro X: pandas DataFrame, features del dataset
        parametro Y: pandas Series, target del dataset
        X, y devono avere la stessa lunghezza, per tale ragione si esegue una verifica.
        Fasi del ciclo:
            - ogni campione i, viene tolto dal training e messo nel test set
            - viene allenato un KNN su N-1 campioni
            - si fa una predizione su i
         
    '''
        # Controllo se X e y sono vuoti
        if self.X.empty or y.empty:
            raise ValueError("X e y non possono essere vuoti.")

        # Controllo che X e y abbiano la stessa lunghezza
        if len(self.X) != len(y):
            raise ValueError("X e y devono avere la stessa lunghezza.")
        
        # Se y è un dataframe lo converto in una Series, per il corretto funzionamento del classificatore
        if isinstance(y, pd.DataFrame):  # Se y è un DataFrame, lo convertiamo in Series
            y = y.iloc[:, 0]
        
        # creiamo un array che conterrà tutti gli X_train, y_train, X_test, y_test
        splits = [] 
        
        for i in range(self.K):
            
            # individuiamo gli INDICI di train/test
            test_index = i  # singolo campione selezionato
            train_indices = [j for j in range(self.K) if j != i]
            
            # dividiamo i DATI in train/test
            X_train = X.iloc[train_indices]  
            y_train = y.iloc[train_indices]
            X_test  = X.iloc[[test_index]]  # Doppie parentesi per mantenere DataFrame
            y_test  = y.iloc[test_index]

            # accumuliamo i training set e i test set
            splits.append((X_train, y_train, X_test, y_test))

        return splits

    def run(self,X,y,k=3):
            
        splits = self.splitter(self.X,y)
        #inizializziamo due array vuoti che conterrano i valori predetti dal modello e quelli reali
        all_actual_values=[]
        all_predicted_values=[]
            
        knn = classificatore_KNN(k=k) # si inizializza il KNN
        for X_train, y_train, X_test, y_test in splits:
            # si allena il modello sul training set specifico per l'iterazione a cui ci troviamo
            knn.fit(X_train, y_train)
            # si fa una predizione su y
            y_pred = knn.predict(X_test) # si nota che poiché X_test ha 1 campione, y_pred avrà 1 elemento
            # si salvano in due variabili il valore corretto e quello predetto dal modello
            actual_value = y_test
            predicted_value = y_pred
    
            # Accumuliamo i valori reali e predetti nei due vettori creati
            all_actual_values.append(actual_value)
            all_predicted_values.append(predicted_value)
        
        return all_actual_values, all_predicted_values
            
    def evaluate(self, X,y,k=3):       
        """
            Metodo per valutare le prestazioni del modello KNN utilizzando il metodo Random Subsampling.

            :param X: pandas DataFrame, rappresenta le feature del dataset.
            :param y: pandas Series, rappresenta le etichette di classe del dataset.
            :param k: int, numero di vicini da considerare per il classificatore KNN (default = 3).

            :return: dict, dizionario contenente le metriche di valutazione del modello.
        """

        # Eseguiamo il metodo run per ottenere i valori reali (actual) e predetti (predicted)
        actual_value, predicted_value = self.run(self.X,y,k)

        # Assicuriamoci che actual_value sia una lista di numeri, non una lista di scalari separati
        actual_value = np.array(actual_value).ravel()

        # Convertiamo predicted_value in un array numpy pulito
        predicted_value = np.array([float(y.iloc[0]) if isinstance(y, pd.Series) else float(y) for y in predicted_value])
        
        # creiamo un'istanza della classe che calcola le metriche di valutazione del modello
        Calculator = metrics_calculator()
        
        # calcoliamo la matrice di confusione
        matrix = Calculator.confusion_matrix(predicted_value, actual_value)
        
        # calcoliamo le metriche
        metriche = Calculator.metrics_evalutation(matrix, predicted_value, actual_value)
        
        return metriche

    '''controllo predizioni:
    print("Distribuzione delle predizioni:")
    print(pd.Series(all_predicted_values).value_counts())
    print("Distribuzione delle etichette reali:")
    print(pd.Series(all_actual_values).value_counts())
    '''

"""
Questo modulo implementa il metodo di Random Subsampling per la valutazione di un modello di classificazione.

Il Random Subsampling consiste nel suddividere ripetutamente un dataset in training e test set in modo casuale,
eseguendo un numero predefinito di iterazioni. 
In ogni iterazione, il modello viene addestrato e testato,
e la prestazione viene misurata per ottenere una valutazione complessiva (data dalla media delle prestazioni singole).
"""

class RandomSubsampling(ModelSelection):
    
    def __init__(self, test_size=0.3, num_splits=10):
        """
        Metodo costruttore per il random subsampling.
        
        test_size: percentuale del dataset usata per il test set.
        num_splits: numero di iterazioni per la suddivisione casuale.
        """

        # Controllo della validità di test_size
        if not 0 < test_size < 1:
            raise ValueError("test_size deve essere compreso tra 0 e 1.")
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
        splits = self.splitter(X, y)
        
        for X_train, X_test, y_train, y_test in splits:

            # inizializzazione e addestramento del modello KNN per ogni split
            knn = classificatore_KNN(k=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            # Calcolo dell'accuratezza basata sul numero di predizioni corrette
            #correct = sum(y_pred == y_test)  # Conta il numero di predizioni corrette
            #accuracies.append(correct / len(y_test))  # Calcola l'accuratezza come rapporto tra corrette e totale

            # accumulo i valori all'interno dei due array
            actual_value.append(y_test)
            predicted_value.append(y_pred)
        
        return actual_value, predicted_value

    def evaluate(self,X,y,k=3):
        """
        Metodo per valutare le prestazioni del modello KNN utilizzando il metodo Random Subsampling.

        :param X: pandas DataFrame, rappresenta le feature del dataset.
        :param y: pandas Series, rappresenta le etichette di classe del dataset.
        :param k: int, numero di vicini da considerare per il classificatore KNN (default = 3).

        :return: dict, dizionario contenente le metriche di valutazione del modello.
        """
        # Eseguiamo il metodo run per ottenere i valori reali (actual) e predetti (predicted)
        actual_value, predicted_value = self.run(X,y,k)
        
        # Convertiamo le liste di pandas Series in array numpy monodimensionali
        actual_value = np.concatenate(actual_value).ravel()
        predicted_value = np.concatenate(predicted_value).ravel()
        
        # creiamo un'istanza della classe che calcola le metriche di valutazione del modello
        Calculator = metrics_calculator()
        
        # calcoliamo la matrice di confusione
        matrix = Calculator.confusion_matrix(predicted_value, actual_value)
        
        # calcoliamo le metriche
        metriche = Calculator.metrics_evalutation(matrix, predicted_value, actual_value)
        
        return metriche

class ModelEvaluationFactory:
    @staticmethod
    def get_validation_strategy(choice, X):
        """
        Factory per restituire l'istanza della strategia di validazione scelta.

        :param choice: int, numero modello scelto ('holdout', 'leave_one_out', 'random_subsampling')

        :return: Istanza della classe di validazione corrispondente, in base alla scelta dell'utente.
        """

        if choice == "1":
            test_size = float(input("Inserisci il valore percentuale che rappresenta la dimensione del test set rispetto all'intero set di dati: "))
            return Holdout(test_size)
        elif choice == "2":
            K = int(input(f"Inserisci il numero di esperimenti (intero) tra 1 e {len(X)}: "))
            return LeaveOneOut(K, X)  
        elif choice == "3":
            test_size = float(input("Inserisci il valore percentuale che rappresenta la dimensione del test set rispetto all'intero set di dati: "))
            num_splits = int(input("Inserisci il numero di splits da realizzare nel metodo: "))
            return RandomSubsampling(test_size, num_splits)
        else:
            raise ValueError(f"Metodo di validazione non riconosciuto.")