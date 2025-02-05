import unittest
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from metrics.metrics_calculator import metrics_calculator
from unittest.mock import patch

class TestMetricsCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = metrics_calculator()

    # TEST CONFUSION MATRIX
    def test_confusion_matrix_basic(self):
        """
        Test standard della matrice di confusione con classe positiva = 2.
        """
        actual = np.array([2, 2, 4, 2, 4])
        predicted = np.array([2, 4, 4, 2, 2])

        cm = self.calc.confusion_matrix(predicted, actual)

        self.assertEqual(cm["TP"], 2)
        self.assertEqual(cm["TN"], 1)
        self.assertEqual(cm["FP"], 1)
        self.assertEqual(cm["FN"], 1)

    def test_confusion_matrix_wrong_dimensions(self):
        """
        Verifica che venga sollevato un ValueError se actual_value e predicted_value hanno dimensioni diverse.
        """
        actual = np.array([2, 2, 4])
        predicted = np.array([2, 2])  # dimensioni diverse
        with self.assertRaises(ValueError):
            self.calc.confusion_matrix(predicted, actual)

    def test_confusion_matrix_series_input(self):
        """
        Verifica il funzionamento con input di tipo Series.
        """
        actual = pd.Series([2, 4, 2])
        predicted = pd.Series([2, 2, 4])
        cm = self.calc.confusion_matrix(predicted, actual)

        self.assertEqual(cm["TP"], 1)
        self.assertEqual(cm["TN"], 0)
        self.assertEqual(cm["FP"], 1)
        self.assertEqual(cm["FN"], 1)

    # TEST METRICS EVALUATION 
    def test_metrics_evaluation_no_auc(self):
        """
        Testa il calcolo delle metriche di base senza AUC.
        """
        cm = {"TP": 2, "TN": 1, "FP": 1, "FN": 1}
        metrics = self.calc.metrics_evalutation(cm)

        self.assertAlmostEqual(metrics["accuracy"], 0.6)
        self.assertAlmostEqual(metrics["error rate"], 0.4)
        self.assertAlmostEqual(metrics["sensitivity"], 2/3, places=5)
        self.assertAlmostEqual(metrics["specificity"], 0.5, places=5)
        self.assertAlmostEqual(metrics["geometric mean"], np.sqrt((2/3)*0.5), places=5)
        self.assertIsNone(metrics["area under the curve"])

    def test_metrics_evaluation_with_auc(self):
        """
        Testa il calcolo di tutte le metriche con AUC.
        """
        actual_value = np.array([2, 2, 4, 4])
        predicted_value = np.array([2, 2, 4, 4])
        predicted_scores = np.array([0.9, 0.8, 0.2, 0.1])  # Probabilità previste

        cm_dict = self.calc.confusion_matrix(predicted_value, actual_value)
        self.assertEqual(cm_dict, {"TP": 2, "TN": 2, "FP": 0, "FN": 0})

        metrics = self.calc.metrics_evalutation(cm_dict, predicted_value=predicted_value ,actual_value=actual_value, predicted_score=predicted_scores)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["error rate"], 0.0)
        self.assertEqual(metrics["sensitivity"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["geometric mean"], 1.0)
        self.assertEqual(metrics["area under the curve"], 1.0)

    # TEST COMPUTE AUC
    def test_compute_auc_single_class_in_actual(self):
        """
        Se actual_value ha una sola classe => AUC=0.
        """
        actual = np.array([2, 2, 2, 2])
        predicted_scores = np.array([0.7, 0.8, 0.6, 0.9])

        auc = self.calc.compute_auc(actual, predicted_scores)
        self.assertEqual(auc, 0)

    def test_compute_auc_single_class_in_predicted(self):
        """
        Se predicted_scores ha un solo valore => AUC=0.
        """
        actual = np.array([2, 2, 4, 4])
        predicted_scores = np.array([0.5, 0.5, 0.5, 0.5])

        auc = self.calc.compute_auc(actual, predicted_scores)
        self.assertEqual(auc, 0)

    def test_compute_auc_perfect_classification(self):
        """
        Caso di classificazione perfetta: AUC=1.0.
        """
        actual = np.array([2, 2, 4, 4])
        predicted_scores = np.array([0.9, 0.8, 0.2, 0.1])

        auc = self.calc.compute_auc(actual, predicted_scores)
        self.assertEqual(auc, 1.0)

    def test_compute_auc_imperfect_classification(self):
        """
        Caso di classificazione imperfetta: vogliamo che l'AUC sia tra 0 e 1.
        """
        actual = np.array([2, 2, 4, 4])
        predicted_scores = np.array([0.8, 0.4, 0.6, 0.2])

        auc = self.calc.compute_auc(actual, predicted_scores)
        self.assertGreater(auc, 0)
        self.assertLess(auc, 1)

    # TEST STAMPA METRICHE (con mock input)
    @patch("builtins.input", return_value="tutte")
    def test_stampa_metriche_all(self, mock_input):
        """
        Testa scegli_e_stampa_metriche simulando l'input "tutte".
        """
        metriche = {
            "accuracy": 0.9,
            "error rate": 0.1,
            "sensitivity": 1,
            "specificity": 1,
            "geometric mean": 1,
            "area under the curve": 0.9
        }
        metriche_filtrate = self.calc.scegli_e_stampa_metriche(metriche)

        self.assertEqual(metriche_filtrate, metriche)

    @patch("builtins.input", return_value="accuracy, sensitivity")
    def test_stampa_metriche_partial(self, mock_input):
        """
        Testa scegli_e_stampa_metriche simulando l'input di alcune metriche.
        """
        metriche = {
            "accuracy": 0.9,
            "error rate": 0.1,
            "sensitivity": 1.0
        }
        metriche_filtrate = self.calc.scegli_e_stampa_metriche(metriche)

        self.assertEqual(metriche_filtrate, {"accuracy": 0.9, "sensitivity": 1.0})


if __name__ == '__main__':
    unittest.main()