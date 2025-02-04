import numpy as np
import pandas as pd
import matplotlib as plt
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
        #if predicted_value is not None and actual_value is not None:
        #    auc_roc_2 = float(self.compute_auc_from_roc(actual_value, predicted_value))  # Passiamo liste
      
        metrics = {
            "accuracy": AR,
            "error rate": ER,
            "sensitivity": TPR,
            "specificity": TNR,
            "geometric mean": g_mean,
            "area under the curve": auc_roc,
            #"area under the curve 2": auc_roc_2
        }

        return metrics
    
    def compute_auc(self, actual_value, predicted_value):
        """
        Calcola l'Area Under the Curve (AUC) della curva ROC, assumendo '2' come classe positiva.

        :param actual_value: np.ndarray o lista, contiene i valori reali delle classi (2 o 4).
        :param predicted_value: np.ndarray o lista, contiene valori predetti (2 o 4) 
        :return: float, valore AUC-ROC.
        """

        positive_label = 2  # Indichiamo la classe positiva (tumore benigno = 2).

        # 1. Controlli preliminari
        # Se c'è una sola classe reale (tutti 2 o tutti 4), l'AUC non ha senso.
        if len(set(actual_value)) < 2:
            print("Errore: Actual values contiene solo una classe, AUC non può essere calcolata!")
            return 0

        # Se il modello produce sempre la stessa previsione, non abbiamo un ranking utile per calcolare la curva ROC.
        if len(set(predicted_value)) < 2:
            print("Errore: Predicted values contiene solo un valore, AUC non può essere calcolata!")
            return 0

        # 2. Costruzione della curva ROC
        # Per calcolare la curva ROC, ci serve un "ranking" dei campioni, in base ai valori di previsione (predicted_value).
        # Otteniamo gli indici che riordinano predicted_value dal più piccolo al più grande.
        sorted_indices = np.argsort(predicted_value)
        # Quindi: sorted_indices[i] = indice del campione che ha l'i-esimo valore più piccolo in predicted_value.

        # Riordiniamo i valori reali (actual_value) secondo questo ordine:
        y_true_sorted = np.array(actual_value)[sorted_indices]
        # Quindi: l'elemento y_true_sorted[0] corrisponde al campione con minore predicted_value, 
        # y_true_sorted[1] a quello con il secondo valore, ecc.
        # In questo modo, scorrendo y_true_sorted, andiamo a costruire i punti della curva ROC secondo soglie crescenti di predicted_value.

        # Calcoliamo la TPR e FPR cumulati, "scorrendo" dal campione col punteggio più basso a quello col punteggio più alto.
        # TPR[i] = frazione di campioni positivi (classe 2) trovati 
        #          tra i primi i+1 campioni (ordinati per predicted_value).
        # FPR[i] = frazione di campioni negativi (classe != 2) trovati 
        #          tra i primi i+1 campioni (ordinati per predicted_value).

        TPR = np.cumsum(y_true_sorted == positive_label) / np.sum(y_true_sorted == positive_label) #True Positive Rate
        FPR = np.cumsum(y_true_sorted != positive_label) / np.sum(y_true_sorted != positive_label) #False Positive Rate

        # Quindi: cumsum(y_true_sorted == 2) conta quanti campioni "veramente positivi" (classe 2) compaiono cumulativamente 
        # mentre avanzano i campioni con predicted_value crescente. 
        # Dividendo per il numero totale di positivi, otteniamo la TPR in ciascun punto.
        # Stessa logica è seguita per il calcolo sui negativi (FPR).

        
        # 3. Calcolo dell'Area Under the Curve
        # Una volta ottenuti i punti (FPR[i], TPR[i]) che definiscono la curva ROC, calcoliamo l'area con un integrale numerico 
        # (metodo dei trapezi).
        auc = np.trapz(TPR, FPR)  # np.trapz(y, x) = integrale di y rispetto a x
        # AUC = area sotto la curva TPR(FPR).

        return auc


    def stampa_metriche(self, metriche):
        """
        Funzione per stampare le metriche scelte dall'utente.
        
        :param metriche: Dizionario contenente le metriche calcolate
        """
        # Chiediamo all'utente quali metriche vuole stampare
        print("\nMetriche disponibili:", ", ".join(metriche.keys()))  # Mostra le metriche disponibili
        scelta = input("Vuoi stampare tutte le metriche o solo alcune? (digita 'tutte' o inserisci i nomi separati da virgola): ").strip().lower() # Eliminiamo spazi bianchi e convertiamo tutte le lettere in minuscolo

        if scelta == "tutte":
            print("Metriche Totali:", metriche)
        else:
            metriche_scelte = [m.strip() for m in scelta.split(",")] # Selezioniamo le metriche da stampare e rimuoviamo gli spazi in eccesso
            metriche_filtrate = {m: metriche[m] for m in metriche_scelte if m in metriche} # Filtriamo le metriche scelte e le inseriamo in un dizionario
            print("Metriche selezionate:", metriche_filtrate)

