import pandas as pd
import numpy as np
import random
from classificatore_KNN import classificatore_KNN
from metrics_calculator import metrics_calculator
#from appoggio import metrics_calculator

#Implementazione del metodo di validazione Leave One Out, che divide per tante volte quanto è il numero di esperimenti richiesto, 
# il dataset in due parti: training set e test set.
# in questo metodo di validazione in ogni iterazione si sceglie casualmente un campione che costituirà il test set, 
# i restanti campioni costituiranno invece il training set. In questo modo avremo la massima percentuale di campioni 
# impiegata per addestrare l'algoritmo.
# Si calcolano inoltre metriche di validazione per il modello.

class LeaveOneOut:
        
    def __init__(self, K: int):
        """
        Metodo costruttore del LeaveOneOut.
        K è il numero di esperimenti che si eseguirà, il suo valore è dato dall'utente
        Per il corretto funzionamento dei metodi successivi è necessario che K sia un intero.
        """
        if not isinstance(K, int):  # Controllo che K sia un intero
            raise ValueError("K deve essere un numero intero.")
        
        self.K = K
        #self.X = X 
    

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
            - si salvano in due variabili il valore corretto e quello predetto dal modello
        
        return: splits, ossia una lista di tuple contenenti X_train, y_train, X_test, y_test
         
    '''
        # Controllo preliminare su K prima di dividere il dataset
        if not 1 <= self.K <= len(X):  # Controlla che K sia valido
            raise ValueError(f"Errore: il numero di esperimenti deve essere compreso tra 1 e {len(X)}.")
        
        # Controllo se X e y sono vuoti
        if X.empty or y.empty:
            raise ValueError("X e y non possono essere vuoti.")

        # Controllo che X e y abbiano la stessa lunghezza
        if len(X) != len(y):
            raise ValueError("X e y devono avere la stessa lunghezza.")
        
        # Se y è un dataframe lo converto in una Series, per il corretto funzionamento del classificatore
        if isinstance(y, pd.DataFrame):  # Se y è un DataFrame, lo convertiamo in Series
            y = y.iloc[:, 0]
        
        # creiamo un array che conterrà tutti gli X_train, y_train, X_test, y_test
        splits = [] 
        # creiamo una variabile che contenga il numero totale di campioni
        n = len(X)  
        # Creaiamo una array di tutti gli indici
        indices = list(range(n))
        # Mescola gli indici in modo casuale
        random.shuffle(indices)

        splits = []
        for i in range(self.K):
            # selezioniamo l'indice di un campione di test, tra quelli del vettore randomizzato 
            test_index = indices[i]
            # creiamo un vettore che contenga tutti gli altri indici
            train_indices = indices[:i] + indices[i+1:]
            '''
            # individuiamo gli INDICI di train/test
            test_index = i  # singolo campione selezionato
            train_indices = [j for j in range(self.K) if j != i]
            '''
            # dividiamo i DATI in train/test
            X_train = X.iloc[train_indices]  
            y_train = y.iloc[train_indices]
            X_test  = X.iloc[[test_index]]  # Doppie parentesi per mantenere DataFrame
            y_test  = y.iloc[test_index]

            # accumuliamo i training set e i test set
            splits.append((X_train, y_train, X_test, y_test))

        return splits

    def run(self,X,y,k=3):
        '''
        Metodo che esegue il classificatore KNN K volte, una per ogni campione.
        parametro X: pandas DataFrame, features del dataset
        parametro Y: pandas Series, target del dataset
        parametro k: int, numero di vicini da considerare per il classificatore KNN (default = 3)

        Fasi del ciclo: 
            - si divide il dataset in training set e test set
            - si allena il modello sul training set specifico per l'iterazione a cui ci troviamo
            - si fa una predizione su y
            - si salvano in due variabili il valore corretto e quello predetto dal modello
            - Accumuliamo i valori reali e predetti nei due vettori creati

        return: all_actual_values, all_predicted_values        
        '''
        # Controllo su K prima di chiamare splitter
        if not 1 <= self.K <= len(X):
            raise ValueError(f"Errore: il numero di esperimenti deve essere compreso tra 1 e {len(X)}.")
        
        splits = self.splitter(X,y)
        #inizializziamo due array vuoti che conterrano i valori predetti dal modello e quelli reali
        all_actual_values=[]
        all_predicted_values=[]
        all_predicted_scores=[]
            
        knn = classificatore_KNN(k=k) # si inizializza il KNN
        for X_train, y_train, X_test, y_test in splits:
            # si allena il modello sul training set specifico per l'iterazione a cui ci troviamo
            knn.fit(X_train, y_train)
            # si fa una predizione su y
            y_pred = knn.predict(X_test) # si nota che poiché X_test ha 1 campione, y_pred avrà 1 elemento
            # si salvano in due variabili il valore corretto e quello predetto dal modello
            
            y_score = knn.predict_proba(X_test)
            
            actual_value = y_test
            predicted_value = y_pred
            predicted_score = y_score
    
            # Accumuliamo i valori reali e predetti nei due vettori creati
            all_actual_values.append(actual_value)
            all_predicted_values.append(predicted_value)
            all_predicted_scores.append(predicted_score)
        
        return all_actual_values, all_predicted_values, all_predicted_scores
            
    def evaluate(self,X,y,k=3):       
        """
            Metodo per valutare le prestazioni del modello KNN utilizzando il metodo Random Subsampling.

            :param X: pandas DataFrame, rappresenta le feature del dataset.
            :param y: pandas Series, rappresenta le etichette di classe del dataset.
            :param k: int, numero di vicini da considerare per il classificatore KNN (default = 3).

            :return: dict, dizionario contenente le metriche di valutazione del modello.
        """
        # Controllo su K prima di procedere con la valutazione
        if not 1 <= self.K <= len(X):
            raise ValueError(f"Errore: il numero di esperimenti deve essere compreso tra 1 e {len(X)}.")
        
        # Eseguiamo il metodo run per ottenere i valori reali (actual) e predetti (predicted)
        actual_value, predicted_value, predicted_score = self.run(X,y,k)

        # Assicuriamoci che actual_value sia una lista di numeri, non una lista di scalari separati
        actual_value = np.array(actual_value).ravel()

        # Convertiamo predicted_value in un array numpy pulito
        predicted_value = np.array([float(y.iloc[0]) if isinstance(y, pd.Series) else float(y) for y in predicted_value])
        
        predicted_score = np.array(predicted_score).ravel()
        
        return actual_value, predicted_value, predicted_score

    

            