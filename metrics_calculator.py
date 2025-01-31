import numpy as np
import pandas as pd
'''
Classe che calcola le metriche di valutazione di un modello di classificazione.

La classe implementa:
1. Matrice di confusione, costituita da True Positive (TP), False Positive (FP), True Negative (TN) e False Negative (FN).
2. Metriche di valutazione, tra cui:
   - Accuracy Rate (AR)
   - Error Rate (ER)
   - Sensitivity (True Positive Rate - TPR)
   - Specificity (True Negative Rate - TNR)
   - Geometric Mean
   - Area Under the Curve (AUC-ROC)
3. Calcolo manuale dell'AUC (area under the curve)

'''
class metrics_calculator():

    def confusion_matrix(self, predicted_value, actual_value):
        """
        Calcola la matrice di confusione dati i valori reali (actual) e quelli predetti (predicted).

        La matrice di confusione serve a valutare le prestazioni di un modello di classificazione binaria.
        Il metodo calcola i seguenti valori:
        - **TP (True Positive)** → Il modello ha predetto positivamente e il valore reale è positivo.
        - **TN (True Negative)** → Il modello ha predetto negativamente e il valore reale è negativo.
        - **FP (False Positive)** → Il modello ha predetto positivamente, ma il valore reale è negativo.
        - **FN (False Negative)** → Il modello ha predetto negativamente, ma il valore reale è positivo.

        :param predicted_value: np.ndarray o pandas Series, valori predetti dal modello.
        :param actual_value: np.ndarray o pandas Series, valori reali del dataset.
        :return: dict, contenente i valori della matrice di confusione {TP, TN, FP, FN}.
        """
        
        positive_class = 2 #positive_class = 2 # prendiamo come caso positivo quello in cui il tumore è benigno
        TP = FP = TN = FN = 0  # Variabili introdotte per il calcolo della Confusion Matrix

        # Se actual_value e predicted_value sono Series, li convertiamo in array numpy
        if isinstance(actual_value, pd.Series):
            actual_value = actual_value.to_numpy()
        if isinstance(predicted_value, pd.Series):
            predicted_value = predicted_value.to_numpy()

        # Controlliamo che le dimensioni siano uguali solo se le variabili sono array
        if isinstance(actual_value, np.ndarray) and isinstance(predicted_value, np.ndarray):
            if actual_value.shape != predicted_value.shape:
                raise ValueError("Le dimensioni di actual_value e predicted_value devono essere uguali.")


        # Calcolo della matrice di confusione
        TP = np.sum((actual_value == positive_class) & (predicted_value == positive_class))
        TN = np.sum((actual_value != positive_class) & (predicted_value != positive_class))
        FP = np.sum((actual_value != positive_class) & (predicted_value == positive_class))
        FN = np.sum((actual_value == positive_class) & (predicted_value != positive_class))
        
        # restituiamo un dizionario con i valori che costituiscono la matrice di confusione  
        return {"TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN }


    def metrics_evalutation(self, confusion_matrix, predicted_value = None, actual_value = None): # i due parametri in ingresso actual e predicted value sono opzionali
        """
        Calcola diverse metriche di valutazione del modello utilizzando la matrice di confusione.

        Le metriche calcolate sono:
        - Accuracy Rate (AR): Percentuale di predizioni corrette sul totale.
        - Error Rate (ER): Percentuale di errori sul totale.
        - Sensitivity (True Positive Rate - TPR): Percentuale di veri positivi correttamente classificati.
        - Specificity (True Negative Rate - TNR): Percentuale di veri negativi correttamente classificati.
        - Geometric Mean (G-Mean): Media geometrica di sensitivity e specificity.
        - AUC-ROC (Area Under the Curve): Misura la capacità del modello di distinguere tra classi.

        :param confusion_matrix: dict, contenente i valori TP, TN, FP, FN.
        :param predicted_value: np.ndarray o pandas Series, valori predetti dal modello (opzionale, serve per AUC).
        :param actual_value: np.ndarray o pandas Series, valori reali del dataset (opzionale, serve per AUC).
        :return: dict, contenente le metriche calcolate.
        """
        
        
        # sfruttiamo il metodo precedente per calcolare TP,TN,FP,FN
        TP, TN, FP, FN = confusion_matrix["TP"], confusion_matrix["TN"], confusion_matrix["FP"], confusion_matrix["FN"]
        
        # a questo punto si possono calcolare le metriche di interesse
        N = TP+TN+FP+FN
        AR = float((TP+TN)/N) if N>0 else 0 # Accuracy Rate
        ER = float((FP + FN)/N) if N>0 else 0 # Error Rate
        TPR = float(TP/(TP+FN)) if (TP+FN)>0 else 0 # True Positive Rate o Sensitivity
        TNR = float(TN/(TN+FP)) if (TN+FP)>0 else 0 # True Negative Rate o Specificity
        g_mean = float(np.sqrt(TPR * TNR)) if (TPR > 0 and TNR > 0) else 0 # Media Geometrica
        auc_roc = None
        if predicted_value is not None and actual_value is not None:
            auc_roc = float(self.compute_auc(actual_value, predicted_value))  # Passiamo liste
      
        metrics = {
            "accuracy": AR,
            "error rate": ER,
            "sensitivity": TPR,
            "specificity": TNR,
            "geometric mean": g_mean,
            "area under the curve": auc_roc
        }

        return metrics

    def compute_auc(self, actual_value, predicted_value):
        """
        Calcola manualmente l'Area Under the Curve (AUC) della curva ROC.

        Viene usato il metodo dell'integrale numerico.
        
        :param actual_value: np.ndarray o lista, valori reali.
        :param predicted_value: np.ndarray o lista, valori predetti.
        :return: float, valore AUC-ROC.
        """

        if len(set(actual_value)) < 2:  # Controlliamo se c'è almeno una classe positiva e una negativa
            print("Errore: Actual values contiene solo una classe, AUC non può essere calcolata!")
            return 0

        if len(set(predicted_value)) < 2:  # Controlliamo se il modello sta facendo sempre la stessa previsione
            print("Errore: Predicted values contiene solo un valore, AUC non può essere calcolata!")
            return 0

        # Calcolo AUC manuale senza sklearn
        sorted_indices = np.argsort(predicted_value)
        y_true_sorted = np.array(actual_value)[sorted_indices]
        y_score_sorted = np.array(predicted_value)[sorted_indices]

        # Calcoliamo le True Positive Rate e False Positive Rate
        TPR = np.cumsum(y_true_sorted == max(y_true_sorted)) / np.sum(y_true_sorted == max(y_true_sorted))
        FPR = np.cumsum(y_true_sorted != max(y_true_sorted)) / np.sum(y_true_sorted != max(y_true_sorted))

        auc = np.trapz(TPR, FPR)  # Calcoliamo l'integrale numerico della curva ROC

        return auc
