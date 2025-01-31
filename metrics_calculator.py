import numpy as np
'''
Classe che calcola la Confusion Matrix dati il valore reale di y e quello predetto.
Si calcolano quindi True Positive (TP), False Positive (FP), True Negative (TN) e False Negative (FN),
al fine di poter calcolare tramite confusion matrix le seguenti metriche: Accuracy Rate (AR), 
Error Rate (ER), Sensitivity e Specificity. 
Si calcola poi la Geometry mean e l'Area under the curve.
'''

class metrics_calculator():

    def confusion_matrix(self, predicted_value, actual_value):

        positive_class = 2 #positive_class = 2 # prendiamo come caso positivo quello in cui il tumore è benigno
        TP = FP = TN = FN = 0  # Variabili introdotte per il calcolo della Confusion Matrix

        if actual_value == positive_class:  # Classe positiva
            if predicted_value == positive_class:
                TP += 1  # Correttamente predetto positivo
            else:
                FN += 1  # Erroneamente predetto negativo

        else:  # Classe negativa
            if predicted_value == positive_class:
                FP += 1  # Erroneamente predetto positivo
            else:
                TN += 1  # Correttamente predetto negativo
        
        # restituiamo un dizionario con i valori che costituiscono la matrice di confusione  
        return {"TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN }


    def metrics_evalutation(self, confusion_matrix, predicted_value = None, actual_value = None): # i due parametri in ingresso actual e predicted value sono opzionali
        # sfruttiamo il metodo precedente per calcolare TP,TN,FP,FN
        TP, TN, FP, FN = confusion_matrix["TP"], confusion_matrix["TN"], confusion_matrix["FP"], confusion_matrix["FN"]
        
        # a questo punto si possono calcolare le metriche di interesse
        N = TP+TN+FP+FN
        AR = (TP+TN)/N if N>0 else 0 # Accuracy Rate
        ER = (FP + FN)/N if N>0 else 0 # Error Rate
        TPR = TP/(TP+FN) if (TP+FN)>0 else 0 # True Positive Rate o Sensitivity
        TNR = TN/(TN+FP) if (TN+FP)>0 else 0 # True Negative Rate o Specificity
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
