import matplotlib.pyplot as plt
from metrics_calculator import metrics_calculator
import pandas as pd
import numpy as np


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

    def save(self, nome_file: str = "model_performance.xlsx") -> None:
        """
        Salva le metriche calcolate in un file Excel.

        Args:     nome_file (str): Nome del file Excel. Default: "model_performance.xlsx".
        """
        if not self.metrics:
            print("Eseguire 'visualizza_metriche()' prima di salvare le metriche.")
            return

        # Salva le metriche in un file Excel
        metrics_df = pd.DataFrame(self.metrics.items(), columns=["Metrica", "Valore"]) #Convertiamo in DataFrame per poi salvare su Excel
        metrics_df.to_excel(nome_file, index=False)
        print(f"Ho salvato le metriche in {nome_file}!")

        """nome_file: nome del file dove salvare dati,
        Non include l'indice del DataFrame nel file Excel,
        """

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
