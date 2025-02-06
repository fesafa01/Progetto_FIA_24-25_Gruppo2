import matplotlib.pyplot as plt
from metrics.metrics_calculator import metrics_calculator
import pandas as pd
import numpy as np
import os
import platform

class metrics_visualizer():

    def __init__(self, actual_value, predicted_value, predicted_score):
        """
        Inizializzare l'oggetto

        Args:
            actual_value, predicted_value: ndarrays contenenti valori reali e predetti
        """
        self.actual_value=actual_value
        self.predicted_value=predicted_value
        self.predicted_score = predicted_score
        self.calculator = metrics_calculator()
        self.matrix = {}
        self.metrics = {}

    def visualizza_metriche(self) -> None:
        """
        Calcola e visualizza tutte le metriche disponibili.
        """
        # calcoliamo la matrice di confusione
        self.matrix = self.calculator.confusion_matrix(self.predicted_value, self.actual_value)
        # calcoliamo le metriche
        self.metrics = self.calculator.metrics_evalutation(self.matrix, self.predicted_value, self.actual_value, self.predicted_score)

        self.metrics= self.calculator.scegli_e_stampa_metriche(self.metrics)


        # Plottare matrice di confusione
        self.plot_conf_matrix()
        # Plottare metriche
        self.plot_metrics()
        # Plottare ROC curve
        tpr, fpr = self.calculator.compute_roc_points(self.actual_value, self.predicted_score)
        self.plot_roc_curve(fpr, tpr)

    def save(self, nome_file: str = "model_performance.xlsx") -> None:
        """
        Salva le metriche calcolate in un file Excel.
        Salva le metriche calcolate in un file Excel, rilevando automaticamente
        se il codice è in esecuzione in Docker, sulla macchina virtuale o su quella locale.    

        Args:  nome_file (str): Nome del file Excel. Default: "model_performance.xlsx".
        
        """
        if not self.metrics:
            print("Eseguire 'visualizza_metriche()' prima di salvare le metriche.")
            return

        # Salva le metriche in un file Excel
        metrics_df = pd.DataFrame(self.metrics.items(), columns=["Metrica", "Valore"]) #Convertiamo in DataFrame per poi salvare su Excel
        
        """
        nome_file: nome del file dove salvare dati,
        Non include l'indice del DataFrame nel file Excel,
        """

        # Identifichiamo se ci troviamo sulla macchina virtuale o sulla macchina locale
        hostname = platform.node()
        if "osboxes" in hostname:  # Se siamo sulla macchina virtuale
            output_dir = "/home/osboxes/Progetto_FIA_24-25_Gruppo2/output"
        elif os.path.exists("/app"):  # Se siamo in Docker
                output_dir = "/app/output"
        else:  # Se siamo sulla macchina locale
                output_dir = os.path.join(os.getcwd(), "output") # Salva nella cartella locale 'output'
        
        output_file = os.path.join(output_dir, nome_file)

        # Controlliamo se la cartella esiste
        if not os.path.exists(output_dir):
            print(f"Creazione della cartella: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        # Salva il file Excel
        try:
            metrics_df.to_excel(output_file, index=False)
            print(f"File salvato con successo in: {output_file}")
        except Exception as e:
            print(f"Errore durante il salvataggio del file: {e}")
            
            '''
            #---------SALVATAGGIO IN SITO REMOTO-------------
            # Definiamo il percorso di salvataggio
            output_dir = "/home/osboxes/Progetto_FIA_24-25_Gruppo2/output"
            output_file = "/output/model_performance.xlsx"

            # Controlliamo se la cartella esiste
            if not os.path.exists(output_dir):
                #print(f"Creazione della cartella: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)

            # Salva il file
            try:
                metrics_df.to_excel(output_file, index=False)
                print(f"File salvato con successo in: {output_file}")
            except Exception as e:
                print(f"Errore durante il salvataggio del file: {e}")
            '''       

    def plot_metrics(self) -> None:
        """
        Plot del grafico a barre e del line chart delle metriche calcolate
        """

        #Grafico a barre delle metriche
        plt.figure(figsize=(10, 6))
        plt.bar(self.metrics.keys(), self.metrics.values(), color=['orange', 'purple', 'red', 'blue', 'cyan', 'green'])
        plt.title("Metriche")
        plt.ylabel("Valore")
        plt.ylim(0, 1)
        plt.xticks(rotation=30)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

        # Crea il line chart
        plt.plot(self.metrics.keys(), self.metrics.values(), marker='o')
        plt.title("Metriche di valutazione")
        plt.xlabel("Metrica")
        plt.ylabel("Valore")
        plt.ylim([0, 1]) 
        plt.xticks(rotation=30)
        plt.grid(True)
        plt.show()
    
    def plot_conf_matrix(self):
        """
        Plot della matrice di confusoone
        """

        confusion_dict=self.matrix

        # 1. Crea la matrice 2x2 (ordinamento: [[TN, FP], [FN, TP]])
        confusion_matrix = np.array([
            [confusion_dict['TN'], confusion_dict['FP']],
            [confusion_dict['FN'], confusion_dict['TP']]
        ])

        # 2. Crea la figura e l'asse
        fig, ax = plt.subplots()

        # 3. Visualizza la matrice con matshow (o imshow) e aggiungi una barra dei colori
        cax = ax.matshow(confusion_matrix, cmap='Blues')
        fig.colorbar(cax)

        # 4. Imposta i tick (etichette degli assi)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted Negative', 'Predicted Positive'])
        ax.set_yticklabels(['Actual Negative', 'Actual Positive'])

        # 5. Aggiungi i valori numerici dentro le celle
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(confusion_matrix[i, j]),
                        va='center', ha='center', color='black')

        plt.title("Matrice di confusione")
        plt.show()

    def plot_roc_curve(self, fpr, tpr):
        '''
        Plotta la curva ROC dato il False Positive Rate (FPR) e il True Positive Rate (TPR).

        :param fpr: np.ndarray, array contenente i valori di False Positive Rate.
        :param tpr: np.ndarray, array contenente i valori di True Positive Rate.
        '''

        # Assicura che la curva inizi a (0,0) e termini a (1,1)
        fpr = np.concatenate(([0], fpr, [1]))
        tpr = np.concatenate(([0], tpr, [1]))

        # Calcolo dell'AUC usando il metodo dei trapezi
        auc_score = np.trapz(tpr, fpr)

        # Plotta la curva ROC
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, marker='o', linestyle='-', label=f"AUC = {auc_score:.3f}", color='blue', linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Classifier (AUC = 0.5)")

        # Etichette e titoli
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True)

        plt.show()
