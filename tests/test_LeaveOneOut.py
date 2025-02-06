import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from validation.LeaveOneOut import LeaveOneOut
from metrics.metrics_visualizer import metrics_visualizer

class TestLeaveOneOut(unittest.TestCase):
    
    def setUp(self):
        """Inizializza un dataset di esempio per i test."""
        self.X = pd.DataFrame({
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10)
        })
        self.y = pd.Series(np.random.choice([2, 4], size=10))  # Due classi
        self.K = len(self.X)  # Usa il massimo possibile per LeaveOneOut
        self.loo = LeaveOneOut(self.K)


    def test_invalid_K(self):
        """Verifica che la classe lanci un errore se K non è valido quando si usano i metodi della classe"""
        
        loocv = LeaveOneOut(100)  # Creiamo un'istanza con K fuori limite
        with self.assertRaises(ValueError):  
            loocv.splitter(self.X, self.y)  # il metodo deve sollevare l'errore perché K > len(X)

        loocv = LeaveOneOut(0)  # Istanza con K = 0 (non valido)
        with self.assertRaises(ValueError):  
            loocv.run(self.X, self.y, choice_distance=1, k=3)  # Deve sollevare l'errore perché K < 1

        loocv = LeaveOneOut(-5)  # Istanza con K negativo
        with self.assertRaises(ValueError):  
            loocv.evaluate(self.X, self.y, choice_distance=1,k=3)  # Deve sollevare l'errore perché K < 1
    
    def test_valid_K(self):
        """Verifica che LeaveOneOut funzioni correttamente con un valore valido di K."""
        loocv = LeaveOneOut(K=len(self.X))  # K uguale alla lunghezza del dataset
        splits = loocv.splitter(self.X, self.y)
        self.assertEqual(len(splits), len(self.X))  # Dovrebbe generare N suddivisioni
    
    def test_invalid_dataset(self):
        """Verifica che venga lanciato un errore se X o y sono vuoti."""
        loocv = LeaveOneOut(K=5)
        empty_X = pd.DataFrame()
        empty_y = pd.Series()
        
        with self.assertRaises(ValueError):
            loocv.splitter(empty_X, self.y)  # X vuoto
        with self.assertRaises(ValueError):
            loocv.splitter(self.X, empty_y)  # y vuoto
        
        with self.assertRaises(ValueError):
            loocv.run(empty_X, self.y, choice_distance=1, k=3)  # X vuoto
        with self.assertRaises(ValueError):
            loocv.run(self.X, empty_y, choice_distance=1, k=3)  # y vuoto
        
        with self.assertRaises(ValueError):
            loocv.evaluate(empty_X, self.y, choice_distance=1,k=3)  # X vuoto
        with self.assertRaises(ValueError):
            loocv.evaluate(self.X, empty_y, choice_distance=1,k=3)  # y vuoto
        
    def test_splitter(self):
        """Verifica che il metodo splitter restituisca il numero corretto di suddivisioni."""
        splits = self.loo.splitter(self.X, self.y)
        self.assertEqual(len(splits), self.K)
        for X_train, y_train, X_test, y_test in splits:
            self.assertEqual(len(X_train), len(self.X) - 1)  # Training set deve avere K-1 campioni
            self.assertEqual(len(X_test), 1)  # Test set deve avere esattamente un campione
        
    def test_run(self):
        """Verifica che il metodo run restituisca liste di lunghezza corretta (K)."""
        actual_value, predicted_value, score = self.loo.run(self.X, self.y, choice_distance=1, k=3)
        self.assertEqual(len(actual_value), self.K)
        self.assertEqual(len(predicted_value), self.K)
        self.assertEqual(len(score), self.K)


    def test_evaluate_output_lengths(self):
        """
        Verifica:
        1) Che actual_value e predicted_value abbiano la stessa lunghezza
        2) Che tale lunghezza corrisponda a K
        """
        actual_values, predicted_values, score = self.loo.evaluate(self.X, self.y, choice_distance=1, k=3)

        # 1) Stessa lunghezza
        self.assertEqual(len(actual_values), len(predicted_values),"I valori actual e predicted devono avere la stessa lunghezza.")
        
        # 2) Lunghezza pari a K
        self.assertEqual(len(actual_values), self.K,
                         f"La lunghezza di actual_values dovrebbe essere {self.K}, invece è {len(actual_values)}.")

    def test_evaluate_raises_value_error(self):
        """
        Verifica che venga sollevato ValueError se:
        - K è fuori range (1 <= K <= len(X))
        - X e y sono vuoti
        - X e y hanno lunghezze diverse
        """
        # 1) K fuori range (ad esempio K=0)
        with self.assertRaises(ValueError):
            LeaveOneOut(K=0).evaluate(self.X, self.y, choice_distance=1, k=3)
        
        # 2) K maggiore della lunghezza di X (ad es. 20 > 10)
        with self.assertRaises(ValueError):
            LeaveOneOut(K=20).evaluate(self.X, self.y, choice_distance=1, k=3)

        # 3) X e y vuoti
        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=int)
        with self.assertRaises(ValueError):
            LeaveOneOut(K=1).evaluate(empty_X, empty_y, choice_distance=1, k=3)

        # 4) X e y di lunghezze diverse
        mismatched_y = self.y.iloc[:-1]  # L'ultimo elemento rimosso
        
        with self.assertRaises(ValueError):
            self.loo.evaluate(self.X, mismatched_y, choice_distance=1, k=3)

if __name__ == '__main__':
    unittest.main()
