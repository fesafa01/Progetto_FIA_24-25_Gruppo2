import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from validation.Holdout import Holdout
import numpy as np
import pandas as pd

class TestHoldout(unittest.TestCase):

    def setUp(self):
        '''
        Creazione di un oggetto Holdout e di un dataset di esempio
        '''
        self.holdout = Holdout(0.3)
        self.X = pd.DataFrame(np.random.rand(10, 5))
        self.y = pd.Series(np.random.randint(0, 2, 10))

    def test_splitter(self):
        '''
        test per il metodo splitter, verifichiamo la corretta suddivisione in train e test set con test_size di default
        '''
        X_train, y_train, X_test, y_test = self.holdout.splitter(self.X, self.y)
        self.assertEqual(len(X_train), 7)
        self.assertEqual(len(y_train), 7)
        self.assertEqual(len(X_test), 3)
        self.assertEqual(len(y_test), 3)

    def test_splitter_test_size(self):
        '''
        test per il metodo splitter, verifichiamo la corretta suddivisione in train e test set con test_size assegnato
        '''
        self.holdout.test_size = 0.5
        X_train, y_train, X_test, y_test = self.holdout.splitter(self.X, self.y)
        self.assertEqual(len(X_train), 5)
        self.assertEqual(len(y_train), 5)
        self.assertEqual(len(X_test), 5)
        self.assertEqual(len(y_test), 5)

    def test_splitter_empty_X(self):
        '''
        Controlliamo che X vuoto sollevi un'eccezione
        '''
        self.X = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.holdout.splitter(self.X, self.y)

    def test_splitter_empty_y(self):
        '''
        Controlliamo che y vuoto sollevi un'eccezione
        '''
        self.y = pd.Series()
        with self.assertRaises(ValueError):
            self.holdout.splitter(self.X, self.y)

    def test_splitter_different_length(self):
        '''
        Controlliamo che se X e y hanno dimensioni diverse si sollevi un'eccezione
        '''
        self.y = pd.Series(np.random.randint(0, 2, 5))
        with self.assertRaises(ValueError):
            self.holdout.splitter(self.X, self.y)
            
        self.holdout.test_size = -1
        with self.assertRaises(ValueError):
            self.holdout.splitter(self.X, self.y)
    
    def test_run(self):
        self.holdout.run(self.X, self.y, 3)
        self.assertTrue(True)

    def test_run_empty_X(self):
        '''
        Controlliamo che X vuoto sollevi un'eccezione
        '''
        self.X = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.holdout.run(self.X, self.y, 3)

    def test_run_empty_y(self):
        '''
        Controlliamo che y vuoto sollevi un'eccezione
        '''
        self.y = pd.Series()
        with self.assertRaises(ValueError):
            self.holdout.run(self.X, self.y, 3)

    def test_evaluate_returned_lengths(self):
        """
        Verifica che:
        1. Il metodo evaluate ritorni due oggetti con la stessa lunghezza.
        2. La lunghezza corrisponda effettivamente al test set.
        """
        # Richiamiamo il metodo evaluate
        actual, predicted = self.holdout.evaluate(self.X, self.y, k=3)
        
        # 1) Verifichiamo che abbiano la stessa lunghezza
        self.assertEqual(len(actual), len(predicted), "Le lunghezze di actual_value e predicted_value dovrebbero coincidere.")

        # 2) Verifichiamo che la lunghezza corrisponda a test_size*len(X)
        expected_test_length = int(len(self.X) * 0.3)
        self.assertEqual(len(actual), expected_test_length,
                         f"La lunghezza del test set dovrebbe essere {expected_test_length}, trovato {len(actual)}.")

    def test_evaluate_raises_value_error(self):
        """
        Verifica che venga sollevato un ValueError:
        - Se X e y sono vuoti
        - Se X e y hanno lunghezze differenti
        """
        # 1) Caso X e y vuoti
        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=int)

        # Verifichiamo che sollevi ValueError su X e y vuoti
        with self.assertRaises(ValueError):
            self.holdout.evaluate(empty_X, empty_y, k=3)

        # 2) Caso X e y di lunghezze diverse
        # Rimuoviamo l'ultimo campione da y, creando una lunghezza differente
        mismatched_y = self.y.iloc[:-1]  # L'ultimo elemento rimosso
        
        with self.assertRaises(ValueError):
            self.holdout.evaluate(self.X, mismatched_y, k=3)

    

if __name__ == '__main__':
    unittest.main()       