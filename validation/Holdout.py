import numpy as np
from classificatore.classificatore_KNN import classificatore_KNN

#Implementazione del metodo di validazione Holdout, che divide il dataset in due parti: training e test set
# in particolare, il test set è una porzione del dataset che non verrà utilizzata per l'addestramento del modello
# ma solo per la sua valutazione

class Holdout:
    '''  
    def __init__(self, test_size=0.3):
        if not 0 < test_size < 1:
            raise ValueError("test_size deve essere compreso tra 0 e 1.")
        self.test_size = test_size
    '''
    def __init__(self, test_size=0.3):
        """
        Costruttore della classe Holdout.
        :param test_size: Può essere:
            - Un valore in [0,1), interpretato come frazione (es. 0.3 = 30%)
            - Un valore in (1,100], interpretato come percentuale (es. 30 = 0.3)
            - Una stringa che termina con '%' (es. "30%")
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

    def run(self, X,y,choice_distance,k=3):
        """
        Addestra e testa un classificatore KNN utilizzando la suddivisione del dataset in training e test set.
        
        :param X: pandas DataFrame, rappresenta le feature del dataset.
        :param y: pandas Series, rappresenta le etichette di classe del dataset.
        :param k: int, numero di vicini da considerare per il classificatore KNN.

       :return: actual_value (numpy array)  
        :return: predicted_value (numpy array)
        :return: predicted_score (numpy array)
        """
        
        
        # chiamiamo il metodo precedente per ottenere X_train, y_train, X_test, y_test
        X_train, y_train, X_test, y_test = self.splitter(X,y)

        # si crea il KNN con il parametro k specifico per l'iterazione a cui ci troviamo
        knn = classificatore_KNN(choice_distance, k)
            
        # si allena il modello sul training set specifico per l'iterazione a cui ci troviamo
        knn.fit(X_train, y_train)
            
        # si fa una predizione su y
        y_pred = knn.predict(X_test) 

        # si calcola lo score
        predicted_score = knn.predict_proba(X_test)

        # si salvano in due variabili il valore corretto e quello predetto dal modello
        actual_value = y_test
        predicted_value = y_pred

        return actual_value, predicted_value, predicted_score

    def evaluate(self, X,y,choice_distance,k=3):
        """
        Valuta le prestazioni del classificatore KNN utilizzando il metodo Holdout.

        :param X: pandas DataFrame, rappresenta le feature del dataset.
        :param y: pandas Series, rappresenta le etichette di classe del dataset.
        :param k: int, numero di vicini da considerare per il classificatore KNN.

        :return: actual_value, numpy array contentente i valori reali.
        :return: predicted_value, numpy array contentente i valori predetti.
        :return: predicted_score, numpy array contentente gli score predetti.
        """
        
        # Eseguiamo il metodo run per ottenere i valori reali (actual) e predetti (predicted)
        actual_value, predicted_value, predicted_score = self.run(X,y,choice_distance,k)
    
        return actual_value, predicted_value, predicted_score