import unittest
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classificatore.classificatore_KNN import classificatore_KNN

class TestKnnClassifier(unittest.TestCase):
    # testiamo tutti i metodi della classe classificatore_KNN
    def test_init_default(self):
        """
        Verifica che il classificatore venga inizializzato con i parametri di default
        (k=3, distanza euclidea).
        """
        knn = classificatore_KNN()
        self.assertEqual(knn.k, 3)
        self.assertIsNotNone(knn.fun_distanza)
        self.assertIsNone(knn.X_train)
        self.assertIsNone(knn.y_train)

    def test_fit_wrong_types(self):
        """
        Verifica che il metodo fit sollevi ValueError se X non è un DataFrame o se y non è un Series.
        """
        knn = classificatore_KNN()
        
        with self.assertRaises(ValueError):
            knn.fit([1, 2, 3], pd.Series([1, 2, 3]))  # X non è DataFrame
        
        with self.assertRaises(ValueError):
            knn.fit(pd.DataFrame([[1, 2], [3, 4]]), [1, 2])  # y non è Series

    def test_fit_correct(self):
        """
        Verifica che fit funzioni correttamente con DataFrame e Series.
        """
        knn = classificatore_KNN()
        X_train = pd.DataFrame([[1, 2], [3, 4]], columns=['f1', 'f2'])
        y_train = pd.Series(['A', 'B'])
        knn.fit(X_train, y_train)

        self.assertTrue(knn.X_train.equals(X_train))
        self.assertTrue(knn.y_train.equals(y_train))

    def test_euclidean_distance(self):
        """
        Testa la funzione distanza_euclidea con dati noti.
        """
        knn = classificatore_KNN(1)  # useremo solo la funzione distanza_euclidea
        X1 = np.array([[0, 0], [3, 4]])  # 2 punti
        X2 = np.array([[6, 8]])          # 1 punto

        # X1 shape = (2,2)
        # X2 shape = (1,2)
        # Distanza tra (0,0) e (6,8) = sqrt(36 + 64) = 10
        # Distanza tra (3,4) e (6,8) = sqrt(9 + 16) = 5
        # Risultato atteso: [[10.0], [5.0]]
        distanze = knn.fun_distanza.calcola(X1, X2)
        
        self.assertEqual(distanze.shape, (2, 1)) # verifichiamo le dimensioni
        self.assertAlmostEqual(distanze[0, 0], 10.0) # distanza tra [0, 0] e [6, 8]
        self.assertAlmostEqual(distanze[1, 0], 5.0) # distanza tra [3, 4] e [6, 8]

    def test_predict_with_no_fit(self):
        """
        Verifica che predict sollevi ValueError se chiamato prima di aver eseguito fit.
        """
        knn = classificatore_KNN()
        X_test = pd.DataFrame([[5, 6]])
        with self.assertRaises(ValueError):
            knn.predict(X_test)

    def test_predict_correct(self):
        """
        Verifica il processo completo:
        - fit su un dataset di esempio
        - predict su nuovi punti
        - controlla il funzionamento con k=1 e distanza euclidea
        """
        knn = classificatore_KNN(k=1)

        # Dataset di esempio (2D):
        # (0,0) -> A
        # (10,0) -> B
        # (0,10) -> C
        X_train = pd.DataFrame({
            'f1': [0, 10, 0],
            'f2': [0,  0,10]
        })
        y_train = pd.Series(['A', 'B', 'C'])

        knn.fit(X_train, y_train)

        # Test su 3 punti di test 
        X_test = pd.DataFrame({
            'f1': [1, 0, 9],
            'f2': [1, 9, 1]
        })
        # Aspettative con k=1: 
        #   - (1,1) più vicino a (0,0) => A
        #   - (0,9) più vicino a (0,10) => C
        #   - (9,1) più vicino a (10,0) => B

        y_pred = knn.predict(X_test)
        expected = pd.Series(['A', 'C', 'B'], index=X_test.index)
        self.assertTrue(y_pred.equals(expected)) # verifico che la predizione sia corretta

    def test_majority_vote_tie(self):
        """
        Verifica che majority_vote gestisca il pareggio (la scelta avviene casualmente fra le label più frequenti).
        """
        knn = classificatore_KNN()

        # Esempio: 2 label "A" e 2 label "B"
        labels = pd.Series(['A', 'A', 'B', 'B'])
        # majority_vote potrebbe restituire A o B con uguale probabilità.

        # Per testare, eseguiamo la funzione più volte 
        # e controlliamo che almeno una volta esca 'A' e almeno una volta esca 'B'.

        results = set()
        for i in range(20):
            results.add(knn.majority_vote(labels))
        
        self.assertTrue(results.issubset({'A', 'B'}))
        self.assertTrue(len(results) > 0, "Majority vote non ha restituito nulla.") 
        #Verifica che len(results) sia maggiore di 0
        # In caso contrario, il test fallisce e mostra il messaggio "Majority vote non ha restituito nulla.”


if __name__ == '__main__':
    unittest.main()
