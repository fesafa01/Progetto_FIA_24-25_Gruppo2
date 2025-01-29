import pandas as pd
import numpy as np
from classificatore_KNN import classificatore_KNN

#Implementazione del metodo di validazione Leave One Out, che divide per tante volte quanto è il numero dei campioni
# il dataset in due parti: training set e test set.
# in questo metodo di validazione in ogni iterazione si sceglie casualmente un campione che costituirà il test set, 
# i restanti campioni costituiranno invece il training set. In questo modo avremo la massima percentuale di campioni 
# impiegata per addestrare l'algoritmo.

class leave_one_out:
        
    def __init__(self, X, y, k=3):
        """
        Metodo costruttore del leave_one_out, con cui salviamo i dati e il k per la KNN.
        X, y devono essere array di uguale lunghezza, per tale ragione si esegue una verifica.
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

    '''
    Metodo che  divide il dataset in training set e test set N volte, dove N è la dimensione di X.
    parametro X: pandas DataFrame, features del dataset
    parametro Y: pandas Series, target del dataset
    Fasi del ciclo:
         - ogni campione i, viene tolto dal train e messo nel test set
         - viene allenato un KNN su N-1 campioni
         - si fa una predizione su i
         - si calcola quanti test sono corretti
    '''
    def split_and_run (self):
        
        N = len(self.X) # Individuo il numero di campioni
        correct = 0  # contatore di test corretti

        for i in range(N):
            
            # individuiamo gli INDICI di train/test
            test_index = i  # singolo campione selezionato
            train_indices = [j for j in range(N) if j != i]
            
            
            # dividiamo i DATI in train/test
            X_train = self.X.iloc[train_indices].values.tolist()
            y_train = self.y.iloc[train_indices].values.tolist()
            X_test  = [self.X.iloc[test_index].values.tolist()]  
            y_test  = [self.y.iloc[test_index]]


            # si crea il KNN
            knn = classificatore_KNN(k=self.k)
            
            # si allena il modello sul training set specifico per l'iterazione a cui ci troviamo
            knn.fit(X_train, y_train)
            
            # si fa una predizione su y
            y_pred = knn.predict(X_test) # si nota che poiché X_test ha 1 campione, y_pred avrà 1 elemento
            
            # si verifica che la predizione sia corretta, se lo è si incrementa il contatore di predizioni corrette
            if y_pred[0] == y_test[0]:
                correct += 1

            
            # Calcolo dell'accuratezza
            accuracy = correct / N
            return accuracy
