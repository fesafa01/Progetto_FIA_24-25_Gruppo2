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
            auc_roc = float(self.compute_auc_from_roc(actual_value, predicted_value))  # Passiamo liste
        #if predicted_value is not None and actual_value is not None:
        #    auc_roc_2 = float(self.compute_auc_from_roc(actual_value, predicted_value))  # Passiamo liste
      
        metrics = {
            "accuracy": AR,
            "error rate": ER,
            "sensitivity": TPR,
            "specificity": TNR,
            "geometric mean": g_mean,
            "area under the curve": auc_roc,
        }

        return metrics
    
    '''
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
    '''

    def compute_roc_points(self, actual_value, predicted_value):
        """
        Calcola i punti (FPR, TPR) della curva ROC iterando su soglie esplicite.

        :param actual_value: np.ndarray o lista, contiene i valori reali delle classi (2 o 4).
        :param predicted_value: np.ndarray o lista, contiene valori predetti (2 o 4)
        :return: (fpr, tpr) - due numpy array contenenti i valori di FPR e TPR
                            per ciascuna soglia, in ordine crescente di soglia.
        """
        actual = np.array(actual_value)
        predicted = np.array(predicted_value)
        positive_label = 2

        #print("\nDistribuzione delle classi nei valori reali:", np.unique(actual, return_counts=True))
        #print("Distribuzione delle classi nei valori predetti:", np.unique(predicted, return_counts=True))

        
        # definizione delle soglie effettive, 2 e 4
        thresholds = np.array([4,2])  

        #Creiamo un insieme di threshold tra il valore minimo e massimo
        #min_val, max_val = min(predicted), max(predicted)
        #thresholds = np.linspace(min_val, max_val, num=2)
        #print("\nThresholds considerati:", thresholds)  

        # Inizializziamo due array vuoti che conterranno i TPR e FPR
        tpr_list =[]
        fpr_list=[]
        
        for m, thr in enumerate(thresholds, start=1):
            pred_pos = (predicted <= thr)  # Binarizzazione delle predizioni: pred_pos è True se predicted <= thr
            
            TP = np.sum((actual == positive_label) & (pred_pos == True))
            FP = np.sum((actual != positive_label) & (pred_pos == True))
            TN = np.sum((actual != positive_label) & (pred_pos == False))
            FN = np.sum((actual == positive_label) & (pred_pos == False))

            # Calcoliamo FPR e TPR con la loro espressione
            TPR = TP / (TP + FN) if (TP + FN) else 0.0
            FPR = FP / (FP + TN) if (FP + TN) else 0.0

            print(f"[Step {m}] Soglia {thr:.2f}: TP={TP}, FP={FP}, TN={TN}, FN={FN}, TPR={TPR:.3f}, FPR={FPR:.3f}")  
            
            # Accumuliamo i valori calcolati negli array 
            tpr_list.append(TPR)
            fpr_list.append(FPR)

        # Convertiamo in numpy array per comodità
        fpr = np.array(fpr_list)
        tpr = np.array(tpr_list)
        
        return tpr, fpr

    def compute_auc_from_roc(self, actual_value, predicted_value):
        """
        Calcola l'area sotto la curva ROC (AUC)
        integrando con il metodo dei trapezi (np.trapz).
        :param actual_value: np.ndarray o lista, contiene i valori reali delle classi (2 o 4).
        :param predicted_value: np.ndarray o lista, contiene valori predetti (2 o 4)
        :return: float, l'area sotto la curva ROC
        """
        # Chiamiamo il metodo precedente per il calcolo della curva ROC
        tpr, fpr = self.compute_roc_points(actual_value, predicted_value)
        #print("\nFPR:", fpr)  # DEBUG
        #print("TPR:", tpr)    # DEBUG
        if len(fpr) == 0 or len(tpr) == 0:
            print("Errore: Lista FPR o TPR vuota! Controlla il calcolo della ROC.")
            return 0.0
    
        # Ordiniamo i valori di FPR in ordine crescente e riordiniamo anche TPR di conseguenza
        sorted_indices = np.argsort(fpr)  # Trova gli indici che ordinano FPR in ordine crescente
        fpr_sorted = fpr[sorted_indices]  # Riordina FPR
        tpr_sorted = tpr[sorted_indices]  # Riordina TPR nella stessa maniera

        #print("\nFPR ordinato:", fpr_sorted)  #DEBUG
        #print("TPR ordinato:", tpr_sorted)    #DEBUG

        # np.trapz(y, x) = calcola l'integrale di y rispetto a x
        # Quindi trapz(TPR, FPR) calcola area sotto la curva TPR(FPR).
        auc = np.trapz(tpr_sorted, fpr_sorted)
        return auc
    
    '''
    def plot_roc_curve(self, actual_value, predicted_value):
        """
        Traccia la curva ROC usando matplotlib.
        """
        tpr, fpr = self.compute_roc_points(actual_value, predicted_value)
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, marker='o', linestyle='-', label='ROC Curve')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid()
        plt.show()
    '''

    def scegli_e_stampa_metriche(self, metriche):
        """
        Funzione per stampare le metriche scelte dall'utente.
        
        :param metriche: Dizionario contenente le metriche calcolate
        """
        # Chiediamo all'utente quali metriche vuole stampare
        print("\nMetriche disponibili:", ", ".join(metriche.keys()))  # Mostra le metriche disponibili
        scelta = input("Vuoi stampare tutte le metriche o solo alcune? (digita 'tutte' o inserisci i nomi separati da virgola): ").strip().lower() # Eliminiamo spazi bianchi e convertiamo tutte le lettere in minuscolo

        if not scelta or scelta == "tutte": #il not scelta serve per selezionare tutte le metriche se l'utente preme invio senza digitare nulla
            print("Metriche Totali:", metriche)
            metriche_filtrate=metriche # le metriche filtrate coincidono con quelle iniziali
        else:
            metriche_scelte = [m.strip() for m in scelta.split(",")] # Selezioniamo le metriche da stampare e rimuoviamo gli spazi in eccesso
            metriche_filtrate = {m: metriche[m] for m in metriche_scelte if m in metriche} # Filtriamo le metriche scelte e le inseriamo in un dizionario
            print("Metriche selezionate:", metriche_filtrate)
        
        return metriche_filtrate

