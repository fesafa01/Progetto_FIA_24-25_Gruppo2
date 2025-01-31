import pandas as pd
import numpy as np
from classificatore_KNN import classificatore_KNN
from metrics_calculator import metrics_calculator
#from appoggio import metrics_calculator

#Implementazione del metodo di validazione Leave One Out, che divide per tante volte quanto è il numero dei campioni
# il dataset in due parti: training set e test set.
# in questo metodo di validazione in ogni iterazione si sceglie casualmente un campione che costituirà il test set, 
# i restanti campioni costituiranno invece il training set. In questo modo avremo la massima percentuale di campioni 
# impiegata per addestrare l'algoritmo.
# Si calcolano inoltre metriche di validazione per il modello.

class leave_one_out:
        
    def __init__(self, X, y, k=3):
        """
        Metodo costruttore del leave_one_out, con cui salviamo i dati e il k per la KNN.
        X, y devono essere array di uguale lunghezza, per tale ragione si esegue una verifica.
        positive_class indica quale classe consideriamo "positiva" e ci servirà per il calcolo delle
        metriche di validazione
        """
        self.X = X
        self.y = y
        self.k = k
        
        # Controllo se X e y sono vuoti
        if self.X.empty or self.y.empty:
            raise ValueError("X e y non possono essere vuoti.")

        # Controllo che X e y abbiano la stessa lunghezza
        if len(self.X) != len(self.y):
            raise ValueError("X e y devono avere la stessa lunghezza.")
        
        # Se y è un dataframe lo converto in una Series, per il corretto funzionamento del classificatore
        if isinstance(self.y, pd.DataFrame):  # Se y è un DataFrame, lo convertiamo in Series
            self.y = self.y.iloc[:, 0]
    

    '''
    Metodo che  divide il dataset in training set e test set N volte, dove N è la dimensione di X.
    parametro X: pandas DataFrame, features del dataset
    parametro Y: pandas Series, target del dataset
    Fasi del ciclo:
         - ogni campione i, viene tolto dal train e messo nel test set
         - viene allenato un KNN su N-1 campioni
         - si fa una predizione su i
         
    '''
    def split_and_run (self):
        
        N = len(self.X) # Individuo il numero di campioni
        Metrics_calculator = metrics_calculator() # inizializziamo il calcolatore delle metriche di valutazione del modello 
        matrice_totale = {"TP": 0, "TN": 0, "FP": 0, "FN": 0} #matrice in cui si accumulano per ogni iterazione TP,TN,FP,FN

        # creiamo due liste per accumulare tutti i valori reali e predetti della y
        all_actual_values = []
        all_predicted_values = []
        
        for i in range(N):
            
            # individuiamo gli INDICI di train/test
            test_index = i  # singolo campione selezionato
            train_indices = [j for j in range(N) if j != i]
            
            # dividiamo i DATI in train/test
    
            X_train = self.X.iloc[train_indices]  
            y_train = self.y.iloc[train_indices]
            X_test  = self.X.iloc[[test_index]]  # Doppie parentesi per mantenere DataFrame
            y_test  = self.y.iloc[test_index]


            # si crea il KNN
            knn = classificatore_KNN(k=self.k)
            
            # si allena il modello sul training set specifico per l'iterazione a cui ci troviamo
            knn.fit(X_train, y_train)
            
            # si fa una predizione su y
            y_pred = knn.predict(X_test) # si nota che poiché X_test ha 1 campione, y_pred avrà 1 elemento
            
            # si salvano in due variabili il valore corretto e quello predetto dal modello
            actual_value = y_test
            predicted_value = y_pred
        
            # Assicuriamoci che predicted_value sia un valore scalare prima del confronto
            if isinstance(actual_value, pd.Series):  # Se è una Series, estrai il primo valore
                actual_value = actual_value.iloc[0]

            if isinstance(predicted_value, pd.Series):  # Anche per y_pred
                predicted_value = predicted_value.iloc[0]

            # Accumuliamo i valori reali e predetti nei due vettori creati
            all_actual_values.append(actual_value)
            all_predicted_values.append(predicted_value)

            # Calcoliamo a questo punto i TP,TN,FP,FN che costituiscono la matrice di confusione
            Calculator = metrics_calculator()
            matrix = Calculator.confusion_matrix(predicted_value, actual_value)

            # sommiamo la matrice di confusione ottenuta a quelle precedenti (accumuliamo i valori)
            for key in matrice_totale:
                matrice_totale[key] += matrix[key]

        metriche = Metrics_calculator.metrics_evalutation(matrice_totale, all_predicted_values, all_actual_values)
        print("Distribuzione delle predizioni:")
        print(pd.Series(all_predicted_values).value_counts())
        print("Distribuzione delle etichette reali:")
        print(pd.Series(all_actual_values).value_counts())
        
        return metriche