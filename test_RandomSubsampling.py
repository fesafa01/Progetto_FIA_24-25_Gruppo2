import unittest
import pandas as pd
import numpy as np
from RandomSubsampling import RandomSubsampling

"""
File di test per la classe RandomSubsampling.

Questo file utilizza il modulo `unittest` per testare la correttezza del metodo di validazione Random Subsampling.
Vengono testate le seguenti funzionalità:

1. Generazione delle suddivisioni (`splitter`)
   - Controllo che il numero di split sia corretto
   -Verifica che le dimensioni dei dataset siano coerenti con `test_size`
   - Test su dataset molto piccoli
   -Verifica che non ci siano suddivisioni uguali (no sovrapposizione totale)

2. Esecuzione della classificazione (`run`)
   -Controllo che il metodo restituisca due liste della stessa lunghezza
   - Verifica che i valori reali e predetti siano effettivamente popolati

3. Valutazione del modello (`evaluate`)
   - Verifica che il metodo restituisca un dizionario con le metriche attese
   -Controllo della presenza delle metriche principali


"""

class TestRandomSubsampling(unittest.TestCase):

    def setUp(self):
        """
        Metodo che viene eseguito prima di ogni test.
        Qui creiamo un dataset di test e un'istanza della classe RandomSubsampling.
        """
        np.random.seed(42)  # Per la riproducibilità dei risultati
        
        # Creazione di un dataset di esempio
        self.X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
        self.y = pd.Series(np.random.choice([0, 1], size=100))  # Classi binarie
        
        # Inizializza un'istanza di RandomSubsampling con test_size 0.3 e 5 splits
        self.rs = RandomSubsampling(test_size=0.3, num_splits=5)

    def test_generate_splits_valid(self):
        """ Verifica che il metodo splitter generi il numero corretto di suddivisioni con dati validi """
        splits = self.rs.splitter(self.X, self.y)
        self.assertEqual(len(splits), self.rs.num_splits)  # Controlla il numero di split

    def test_generate_splits_default_values(self):
        """ Testa il comportamento con i valori di default (test_size=0.3, num_splits=10) """
        rs_default = RandomSubsampling()
        splits = rs_default.splitter(self.X, self.y)
        self.assertEqual(len(splits), 10)  # Dovrebbe creare 10 splits di default
        test_size_expected = int(len(self.X) * 0.3)
        for _, X_test, _, y_test in splits:
            self.assertEqual(len(X_test), test_size_expected)  # Controlla la dimensione del test set

    def test_generate_splits_small_dataset(self):
        """ Testa il comportamento su un dataset molto piccolo """
        X_small = pd.DataFrame(np.random.rand(5, 3), columns=['A', 'B', 'C'])
        y_small = pd.Series(np.random.choice([0, 1], size=5))
        rs_small = RandomSubsampling(test_size=0.4, num_splits=3)
        splits = rs_small.splitter(X_small, y_small)
        self.assertEqual(len(splits), 3)  # Deve generare esattamente 3 split

    def test_generate_splits_with_invalid_test_size(self):
        """ Verifica che venga sollevato un errore se test_size non è tra 0 e 1 """
        with self.assertRaises(ValueError):
            RandomSubsampling(test_size=1.2)  # Valore non valido
        with self.assertRaises(ValueError):
            RandomSubsampling(test_size=0)  # Valore non valido

    def test_generate_splits_with_invalid_n_iter(self):
        """ Verifica che venga sollevato un errore se num_splits è negativo o zero """
        with self.assertRaises(ValueError):
            RandomSubsampling(num_splits=-1)
        with self.assertRaises(ValueError):
            RandomSubsampling(num_splits=0)

    def test_generate_splits_no_overlap(self):
        """ verifica che i test set generati nelle diverse iterazioni siano diversi.
        Controlla che non ci sia una sovrapposizione totale tra i test set. """
        splits = self.rs.splitter(self.X, self.y)
        test_sets = [set(X_test.index) for _, X_test, _, _ in splits]
        # Se tutti i test set fossero uguali, la lunghezza dell'unione sarebbe la stessa di uno solo
        unique_test_points = set.union(*test_sets)
        self.assertGreater(len(unique_test_points), len(test_sets[0]))  # Controlla se c'è varietà tra i test set

    def test_run(self):
        """
        Verifica il metodo run.
        - Controlla che restituisca due liste della stessa lunghezza.
        - Controlla che i valori reali e predetti siano effettivamente popolati.
        """
        actual_value, predicted_value = self.rs.run(self.X, self.y, k=3)
        
        # Controlla che siano liste della stessa lunghezza
        self.assertEqual(len(actual_value), len(predicted_value))
        
        # Controlla che nessun valore sia None
        for y_actual, y_pred in zip(actual_value, predicted_value):
            self.assertIsNotNone(y_actual)
            self.assertIsNotNone(y_pred)

    def test_evaluate(self):
        """
        Testa il metodo evaluate.
        - Controlla che restituisca un dizionario con le metriche attese.
        - Controlla la presenza delle metriche principali.
        """
        metrics = self.rs.evaluate(self.X, self.y, k=3)
        
        # Controlla che il risultato sia un dizionario
        self.assertIsInstance(metrics, dict)
        
        # Controlla che il dizionario contenga le metriche attese
        expected_keys = {'accuracy', 'error rate', 'sensitivity', 'specificity', 'geometric mean', 'area under the curve'}
        self.assertTrue(expected_keys.issubset(metrics.keys()))

if __name__ == '__main__':
    unittest.main()