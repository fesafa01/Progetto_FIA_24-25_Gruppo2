import unittest
import numpy as np
import pandas as pd
from metrics_calculator import metrics_calculator
from unittest.mock import patch

class TestMetricsCalculator(unittest.TestCase):
    # testiamo tutti i metodi della classe metrics_calculator
    def setUp(self):
        self.calc = metrics_calculator()

    # TEST CONFUSION MATRIX
    def test_confusion_matrix_basic(self):
        """
        Test standard della matrice di confusione. Classe positiva = 2
        """
        actual = np.array([2, 2, 4, 2, 4])     # 3 positivi (valore=2), 2 negativi (valore=4)
        predicted = np.array([2, 4, 4, 2, 2])  # Predizioni miste
        # Qui:
        # TP = (2,2),(2,2) => 2
        # TN = (4,4) => 1
        # FP ovvero actual=4, predicted=2 => 1 (ultimo indice)
        # FN ovvero actual=2, predicted=4 => 1 (secondo indice)
        cm = self.calc.confusion_matrix(predicted, actual)
        
        # verifichiamo che TP,TN,FP,FN abbiano i valori che mi aspetto
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
            a = self.calc.confusion_matrix(predicted, actual)

    def test_confusion_matrix_series_input(self):
        """
        Verifica il funzionamento con input di tipo Series al posto di array numpy.
        """
        actual = pd.Series([2, 4, 2])
        predicted = pd.Series([2, 2, 4])
        cm = self.calc.confusion_matrix(predicted, actual)
        # classe positiva=2
        # actual: pos, neg, pos
        # pred:   pos, pos, neg
        # => TP=1 (primo), TN=0, FP=1 (secondo), FN=1 (terzo)
        self.assertEqual(cm["TP"], 1)
        self.assertEqual(cm["TN"], 0)
        self.assertEqual(cm["FP"], 1)
        self.assertEqual(cm["FN"], 1)

    # TEST METRICS EVALUATION 
    def test_metrics_evaluation_no_auc(self):
        """
        Testa il calcolo delle metriche di base senza AUC 
        (predicted_value e actual_value = None).
        """
        cm = {"TP": 2, "TN": 1, "FP": 1, "FN": 1}  # preso per esempio dal test precedente
        metrics = self.calc.metrics_evalutation(cm)
        # N=5 
        # accuracy= (2+1)/5=3/5=0.6
        # error=2/5=0.4
        # TPR=2/(2+1)=2/3=0.666...
        # TNR=1/(1+1)=0.5
        # g-mean=sqrt(0.6667*0.5) 
        self.assertAlmostEqual(metrics["accuracy"], 0.6)
        self.assertAlmostEqual(metrics["error rate"], 0.4)
        self.assertAlmostEqual(metrics["sensitivity"], 2/3, places=5)
        self.assertAlmostEqual(metrics["specificity"], 0.5, places=5)
        self.assertAlmostEqual(metrics["geometric mean"], np.sqrt((2/3)*0.5), places=5)
        # AUC non essendo calcolata in questo metodo, deve essere None
        self.assertIsNone(metrics["area under the curve"])

    def test_metrics_evaluation_with_auc(self):
        """
        Testa il calcolo di tutte le metriche
        Classe positiva = 2
        """
        # Caso "perfetto", la predizione è esattamente uguale al valore reale => nessun errore
        actual_value = np.array([2, 2, 4, 4])
        predicted_value = np.array([2, 2, 4, 4])

        # Confusion matrix: TP=2 (due 2 corretti), TN=2 (due 4 corretti), FP=0, FN=0
        cm_dict = self.calc.confusion_matrix(predicted_value, actual_value)
        self.assertEqual(cm_dict, {"TP": 2, "TN": 2, "FP": 0, "FN": 0}) # verifica il corretto calcolo della confusion matrix

        # Calcoliamo tutte le metriche
        metrics = self.calc.metrics_evalutation(cm_dict, predicted_value, actual_value)

        # verifichiamo che vengano calcolate correttamente
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["error rate"], 0.0)
        self.assertEqual(metrics["sensitivity"], 1.0)   # TP/(TP+FN)=2/2=1
        self.assertEqual(metrics["specificity"], 1.0)   # TN/(TN+FP)=2/2=1
        self.assertEqual(metrics["geometric mean"], 1.0)

        # AUC=1.0 in un caso perfetto di classificazione binaria
        self.assertIsNotNone(metrics["area under the curve"])
        self.assertEqual(metrics["area under the curve"], 1.0)


    # TEST COMPUTE AUC
    def test_compute_auc_single_class_in_actual(self):
        """
        Se actual_value ha una sola classe => AUC=0
        """
        # actual = tutti 2 (classe positiva), nessun 4 => ho una single class
        actual = [2, 2, 2, 2]
        # predicted = 2 o 4, deve avere almeno 2 classi
        predicted = [2, 4, 4, 4]
        auc = self.calc.compute_auc(actual, predicted)
        self.assertEqual(auc, 0, "AUC dovrebbe essere 0 se actual ha una sola classe.")

    def test_compute_auc_single_class_in_predicted(self):
        """
        Se predicted_value ha un solo valore => AUC=0
        """
        # ho due classi in actual
        actual = [2, 2, 4, 4]
        # ho una sola classe in predicted (tutti 2)
        predicted = [2, 2, 2, 2]
        auc = self.calc.compute_auc(actual, predicted)
        self.assertEqual(auc, 0, "AUC dovrebbe essere 0 se predicted ha una sola classe.")

    def test_compute_auc_perfect_classification(self):
        """
        Caso di classificazione perfetta:
        actual = [2, 2, 4, 4], predicted= [2, 2, 4, 4]
        Ci aspettiamo AUC=1.0 (curva ROC perfetta).
        """
        actual = [2, 2, 4, 4]
        predicted = [2, 2, 4, 4]
        auc = self.calc.compute_auc(actual, predicted)
        self.assertEqual(auc, 1.0, f"AUC dovrebbe essere 1.0 nel caso perfetto, invece è {auc}")

    def test_compute_auc_imperfect_classification(self):
        """
        Caso di classificazione imperfetta: vogliamo che l'AUC sia < 1.
        """
        # 2 positivi (2,2) e 2 negativi (4,4).
        actual = [2, 2, 4, 4]
        predicted = [2, 4, 2, 4]
        # Spiegazione:
        #  - indice 0 => actual=2, predicted=2 (va bene)
        #  - indice 1 => actual=2, predicted=4 (errore, un positivo con "score" alto)
        #  - indice 2 => actual=4, predicted=2 (errore, un negativo con "score" basso)
        #  - indice 3 => actual=4, predicted=4 (va bene)

        auc = self.calc.compute_auc(actual, predicted)
        self.assertGreater(auc, 0, "AUC deve essere >0 in un caso di parziale distinzione.")
        self.assertLess(auc, 1, f"AUC deve essere <1 in un caso non perfetto, ma è {auc}")
        
    def test_compute_auc_partial_classification(self):
        """
        Un altro caso, con 3 esempi classe 2 e 3 esempi classe 4.
        Ho predicted con qualche errore ma non tutti.
        """
        actual = [2, 2, 2, 4, 4, 4]
        # Previsioni con qualche errore
        predicted = [2, 4, 2, 4, 4, 2]  
        auc = self.calc.compute_auc(actual, predicted)
        self.assertTrue(0 < auc < 1, f"AUC fuori range (0,1): {auc}")


    # TEST STAMPA METRICHE (con mock input)
    @patch("builtins.input", return_value="tutte")
    def test_stampa_metriche_all(self, mock_input):
        """
        Testa stampa_metriche simulando l'input "tutte".
        """
        metriche = {
            "accuracy": 0.9,
            "error rate": 0.1,
            "sensitivity": 1,
            "specificity": 1,
            "geometric mean": 1,
            "area under the curve": 0.9
        }
        # Non ci aspettiamo alcun ValueError, stampiamo "Metriche Totali: {...}"
        self.calc.stampa_metriche(metriche)

    @patch("builtins.input", return_value="accuracy, sensitivity")
    def test_stampa_metriche_partial(self, mock_input):
        """
        Testa stampa_metriche simulando l'input di alcune metriche.
        """
        metriche = {
            "accuracy": 0.9,
            "error rate": 0.1,
            "sensitivity": 1.0
        }
        self.calc.stampa_metriche(metriche)
        # Ci aspettiamo "Metriche selezionate: {'accuracy': ..., 'sensitivity': ...}"


if __name__ == '__main__':
    unittest.main()