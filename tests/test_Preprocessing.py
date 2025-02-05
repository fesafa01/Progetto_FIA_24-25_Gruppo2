import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing import Preprocessing
import pandas as pd
import numpy as np

class TestPreprocessing(unittest.TestCase):
# testiamo tutti i metodi della classe Preprocessing
    def test_normalize_column_name(self):
        """
        Testa la funzione normalize_column_name, verificando che sostituisca underscore, trattini e
        converta in minuscolo correttamente.
        """
        self.assertEqual(Preprocessing.normalize_column_name("Clump_Thickness"), "clump thickness")
        self.assertEqual(Preprocessing.normalize_column_name(" Uniformity-OfCellSize ")," uniformity ofcellsize ")
        self.assertEqual(Preprocessing.normalize_column_name("SOME_RANDOM__Value--Test"),"some random  value  test")

    def test_filter_and_reorder_columns(self):
        """
        Testa la funzione filter_and_reorder_columns
        verificando la logica di matching fuzzy e il reorder delle colonne.
        """
        # Costruiamo un DataFrame di esempio con nomi di colonne "imprecisi"
        data = {
            'sample code num': [1, 2],
            'CLUMP thickness': [5, 3],
            'uniformity_cell_SHAPE': [10, 12],
            'Uniformity_of_Cell_Size': [1, 2],
            'bare  nuc': [10, 20],
            'Class': [2, 4],
            'unrelated_column': [99, 99]  # Colonna indesiderata, da eliminare
        }
        df = pd.DataFrame(data)

        # Applichiamo la funzione
        df_filtered = Preprocessing.filter_and_reorder_columns(df, threshold=70)

        # Colonne attese nell'ordine definito in filter_and_reorder_columns
        expected_columns = [
            "Sample code number",
            "Clump Thickness",
            "Uniformity of Cell Shape",
            "Uniformity of Cell Size",
            "Marginal Adhesion",
            "Single Epithelial Cell Size",
            "Bare Nuclei",
            "Bland Chromatin",
            "Normal Nucleoli",
            "Mitoses",
            "Class"
        ]
        self.assertListEqual(list(df_filtered.columns), expected_columns)
        
        # Verifichiamo che le colonne mancanti siano state aggiunte con NaN e che i valori esistenti siano stati mantenuti.
        self.assertTrue(df_filtered["Sample code number"].equals(pd.Series([1, 2], name="Sample code number")))
        # "Bare Nuclei" corrisponde a "bare  nuc" del dataset originale
        self.assertTrue(df_filtered["Bare Nuclei"].equals(pd.Series([10, 20], name="Bare Nuclei")))
        # "Uniformity of Cell Size" corrisponde a "Uniformity_of_Cell_Size" del dataset originale
        self.assertTrue(df_filtered["Uniformity of Cell Size"].equals(pd.Series([1, 2], name="Uniformity_of_Cell_Size")))
        # Le colonne aggiunte (senza match) devono essere NaN
        self.assertTrue(df_filtered["Marginal Adhesion"].isna().all())
        self.assertTrue(df_filtered["Bland Chromatin"].isna().all())
        self.assertFalse(df_filtered["Uniformity of Cell Shape"].isna().all()) # questa colonna che ha un match, non deve essere NaN

    def test_drop_nan(self):
        """
        Testa la funzione drop_nan, che rimuove le righe che non rispettano una soglia minima di valori non NaN.
        """
        data = {
            'A': [1, np.nan, 3],
            'B': [4, 5, np.nan],
            'C': [7, 8, 9]
        }
        df = pd.DataFrame(data)
        # threshold = 2 significa che ogni riga deve avere almeno 2 valori non-NaN
        df_dropped = Preprocessing.drop_nan(df, threshold=2)

        # Riga 0 => (1,4,7) -> tutti valori validi, non viene scartata
        # Riga 1 => (NaN, 5,8) -> 2 valori non NaN, non viene scartata
        # Riga 2 => (3, NaN, 9) -> 2 valori non NaN, non viene scartata
        self.assertEqual(len(df_dropped), 3) # verifico che rimangano tutte e 3 le righe del df

        # Se threshold = 3, deve restare solo la riga 0
        df_dropped_stricter = Preprocessing.drop_nan(df, threshold=3)
        self.assertEqual(len(df_dropped_stricter), 1)
        self.assertTrue(df_dropped_stricter.iloc[0]["A"] == 1) # verifico che ci sia il valore che mi aspetto

    def test_drop_nan_target(self):
        """
        Testa la funzione drop_nan_target, che rimuove le righe con NaN nella colonna target specificata.
        """
        data = {
            'A': [10, 20, 30],
            'Target': [1, np.nan, 3]
        }
        df = pd.DataFrame(data)
        df_cleaned = Preprocessing.drop_nan_target(df, "Target")

        # Deve eliminare la seconda riga (indice 1) perché Target è NaN
        self.assertEqual(len(df_cleaned), 2)
        self.assertEqual(df_cleaned["A"].tolist(), [10, 30]) # verifico che ci siano i valori che mi aspetto in A e in target
        self.assertEqual(df_cleaned["Target"].tolist(), [1, 3])

    def test_nan_handler_mean(self):
        """
        Testa la sostituzione dei valori NaN con la media.
        """
        data = {
            'A': [1, 2, np.nan],
            'B': [4, np.nan, 6]
        }
        df = pd.DataFrame(data)
        df_filled = Preprocessing.nan_handler(df, 'media')

        # Valore medio colonna A = (1 + 2) / 2 = 1.5
        # Valore medio colonna B = (4 + 6) / 2 = 5
        self.assertAlmostEqual(df_filled.loc[2, 'A'], 1.5) # verifico che al posto dei Nan ci siano le medie 
        self.assertAlmostEqual(df_filled.loc[1, 'B'], 5)

        # si usa assrtAlmostEqual per due valori numerici in virgola mobile: chiedo che siano uguali entro un certo margine di tolleranza

    def test_nan_handler_median(self):
        """
        Testa la sostituzione dei valori NaN con la mediana.
        """
        data = {
            'A': [1, 10, np.nan],
            'B': [100, np.nan, 300]
        }
        df = pd.DataFrame(data)
        df_filled = Preprocessing.nan_handler(df, 'mediana')

        # Mediana A = 1, 10 => (1+10)/2 = 5.5
        # Mediana B = 100, 300 => (100+300)/2 = 200
        self.assertAlmostEqual(df_filled.loc[2, 'A'], 5.5) # verifico che al posto dei Nan ci siano le mediane
        self.assertAlmostEqual(df_filled.loc[1, 'B'], 200)

    def test_nan_handler_mode(self):
        """
        Testa la sostituzione dei valori NaN con la moda.
        """
        data = {
            'A': [10, 10, np.nan, 20],  # moda = 10
            'B': [1, 2, 2, np.nan]      # moda = 2
        }
        df = pd.DataFrame(data)
        df_filled = Preprocessing.nan_handler(df, 'moda')

        self.assertEqual(df_filled.loc[2, 'A'], 10) # verifico che al posto dei Nan ci siano le mode
        self.assertEqual(df_filled.loc[3, 'B'], 2)

    def test_nan_handler_remove(self):
        """
        Testa la rimozione delle righe che hanno valori NaN.
        """
        data = {
            'A': [1, np.nan, 3],
            'B': [4, 5, np.nan]
        }
        df = pd.DataFrame(data)
        df_removed = Preprocessing.nan_handler(df, 'rimuovi')
        # Rimuove righe che hanno NaN. 
        # La sola riga 0 (A=1,B=4) non contiene NaN 
        # La riga 1 ha un Nan in A
        # La riga 2 ha un NaN in B
        # Ci aspettiamo che rimanga soltanto 1 riga 
        self.assertEqual(len(df_removed), 1)
        self.assertTrue(df_removed.iloc[0]['A'] == 1)

    def test_nan_handler_invalid_method(self):
        """
        Testa il caso di passaggio di un metodo non valido, che deve sollevare ValueError.
        """
        df = pd.DataFrame({'A': [1, 2]})
        with self.assertRaises(ValueError):
            a = Preprocessing.nan_handler(df, 'invalid') # se non passo media, mediana, moda o rimuovi, solleva un'eccezione

    def test_feature_scaling_minmax(self):
        """
        Testa la normalizzazione Min-Max (method=1).
        """
        data = {
            'A': [0, 10, 20],
            'B': [100, 200, 300]
        }
        df = pd.DataFrame(data)
        df_scaled = Preprocessing.feature_scaling(df, method=1)

        # Per colonna A: min=0, max=20 => valori scalati: [0, 0.5, 1.0]
        # Per colonna B: min=100, max=300 => valori scalati: [0, 0.5, 1.0]
        self.assertAlmostEqual(df_scaled.loc[0, 'A'], 0.0)
        self.assertAlmostEqual(df_scaled.loc[1, 'A'], 0.5)
        self.assertAlmostEqual(df_scaled.loc[2, 'A'], 1.0)

        self.assertAlmostEqual(df_scaled.loc[0, 'B'], 0.0)
        self.assertAlmostEqual(df_scaled.loc[1, 'B'], 0.5)
        self.assertAlmostEqual(df_scaled.loc[2, 'B'], 1.0)

    def test_feature_scaling_zscore(self):
        """
        Testa la standardizzazione Z-score (method=2).
        """
        data = {
            'A': [10, 20, 30, 40],
            'B': [5, 5, 7, 9]
        }
        df = pd.DataFrame(data)
        df_scaled = Preprocessing.feature_scaling(df, method=2)

        # Controlliamo che la media sia circa 0 e std sia circa 1 (entro un certo margine).
        mean_A = df_scaled['A'].mean()
        std_A = df_scaled['A'].std()

        mean_B = df_scaled['B'].mean()
        std_B = df_scaled['B'].std()

        self.assertAlmostEqual(mean_A, 0, delta=1e-7)
        self.assertAlmostEqual(std_A, 1, delta=1e-7)

        self.assertAlmostEqual(mean_B, 0, delta=1e-7)
        self.assertAlmostEqual(std_B, 1, delta=1e-7)

    def test_feature_scaling_invalid_method(self):
        """
        Testa il passaggio di un metodo non valido, che deve sollevare ValueError.
        """
        df = pd.DataFrame({'A': [1, 2]})
        with self.assertRaises(ValueError):
            a = Preprocessing.feature_scaling(df, method=999) # se non passo 1 o 2, solleva un'eccezione

    def test_split(self):
        """
        Testa la funzione split per separare features (X) e target (y),
        mantenendo la prima colonna intatta (ma poi escludendola dal training).
        """
        data = {
            'ID': [101, 102, 103],  # prima colonna
            'Feature1': [1, 2, 3],
            'Feature2': [10, 20, 30],
            'Class': ['A', 'B', 'C']
        }
        df = pd.DataFrame(data)

        X, y = Preprocessing.split(df, target='Class')

        # X deve contenere 'Feature1' e 'Feature2', ma non 'Class' e non la prima colonna
        self.assertTrue('ID' not in X.columns)
        self.assertTrue('Class' not in X.columns)
        self.assertTrue('Feature1' in X.columns)
        self.assertTrue('Feature2' in X.columns)

        # y deve essere la colonna "Class"
        self.assertEqual(list(y), ['A', 'B', 'C'])


if __name__ == '__main__': 
    unittest.main()