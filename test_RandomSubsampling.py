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
        """ Verifica che venga sollevato un errore se test_size è 0 """
        '''with self.assertRaises(ValueError):
            RandomSubsampling(test_size=1.2)  # Valore non valido''' #Pezzo di codice rimosso in quanto è stata aggiunta la funzionalità di inserire un test_size > 1, che viene diviso per 100
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


    def test_evaluate_returned_lengths(self):
        """
        Testa che il metodo evaluate:
         - Ritorni due oggetti (actual_value, predicted_value) della stessa lunghezza
         - Tale lunghezza corrisponda alla somma dei campioni di test prelevati in tutte le suddivisioni
        """
        
        # Richiamiamo il metodo evaluate
        actual, predicted = self.rs.evaluate(self.X, self.y, k=3)
        
        # Verifichiamo che abbiano la stessa lunghezza
        self.assertEqual(len(actual), len(predicted),
                         "Le lunghezze di actual_value e predicted_value dovrebbero coincidere.")

        # Verifichiamo che la lunghezza corrisponda alla somma dei campioni test in tutte le split
        # In ogni split: test_size = 30 campioni
        # Con num_splits=5, i campioni totali in test sono 5 * 30 = 150
        expected_length = int(len(self.X) * 0.3) * 5
        self.assertEqual(len(actual), expected_length,
                         f"Ci si aspetta un totale di {expected_length} campioni nel test set unendo le 5 split.")

    def test_evaluate_raises_value_error(self):
        """
        Testa che venga sollevato ValueError:
         - Se X e y sono vuoti
         - Se X e y hanno lunghezze differenti
        """

        # 1) Caso X e y vuoti
        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=int)
        
        with self.assertRaises(ValueError):
            self.rs.evaluate(empty_X, empty_y, k=3)

        # 2) Caso X e y di lunghezze differenti
        # Eliminiamo un campione da y per creare la discrepanza
        mismatched_y = self.y.iloc[:-1]
        
        with self.assertRaises(ValueError):
            self.rs.evaluate(self.X, mismatched_y, k=3)

if __name__ == '__main__':
    unittest.main()