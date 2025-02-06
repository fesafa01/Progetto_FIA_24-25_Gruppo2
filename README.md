# Progetto_FIA_24-25_Gruppo2

## Descrizione
Spazio di lavoro per lo sviluppo del progetto del corso di Fondamenti di Intelligenza Artificiale dell'Università Campus Bio-Medico di Roma. Il progetto include lo sviluppo di un classificatore k-NN implementato da zero.

Il progetto si concentra sulla **classificazione dei tumori** come benigni o maligni, utilizzando un dataset che contiene informazioni su diverse caratteristiche delle cellule tumorali. L'obiettivo non è solo quello di ottenere un modello prestante dal punto di vista di metriche restituite, ma anche di sviluppare una pipeline generica e flessibile che permetta di eseguire esperimenti e validazioni personalizzate.

## Funzionalità
- Inserimento dinamico del percorso e del nome del file, contenente il dataset.
- Selezione interattiva nella scelta del tipo di distanza da utilizzare nella predizione.
- Selezione interattiva nella scelta del modello da applicare per dividere il dataset ed eventuali parametri caratteristici di questi modelli.
- Holdout: metodo che permette la suddivisione di features e labels in train e test usati per l'addestramento e la valutazione del modello.
- Leave-one-out: metodo che permette di suddivere per tante volte quanto è il numero di campioni, il dataset in training e test set, costituito in ogni iterazione da un solo campione. Il metodo permette di addestrare e testare il classificatore al fine di validarlo.
- Random Subsampling: metodo che suddivide il dataset in training e test set più volte in modo casuale per ottenere una valutazione più stabile delle prestazioni del modello.
- Classificatore k-NN: Un modello che predice le etichette dei dati basandosi sui k nearest neighbours, utilizzando la distanza euclidea (modificabile se necessario).
- Calcolo delle seguenti metriche per ogni metodo di validazione del modello: accuracy rate (AR), error rate (ER), sensitivity (TPR), specificity (TNR), geometric mean, area under the curve. Vengono stampate solo le metriche richieste dall'utente.
Le metriche vengono calcolate in due metodi nel caso del Leave One Out e del Random Subsampling: il primo metodo prevede la stampa delle metriche calcolate come media delle metriche di ogni singola iterazione; il secondo metodo prevede la stampa delle emtriche calcolate aggregando le iterazioni tra di loro, quindi generando un'unica confusion matrix invece che una per ogni iterazione.
Nel caso dell'Holdout, avendo una sola iterazione, i due metodi coicnidono, quindi viene stampato un solo tipo di metriche.
Sono stati mantenuti entrambi i metodi per valutarne le differenze: il metodo che sfrutta la media delle metriche delle iterazioni è da considerare come standard.
- Plot dei grafici relativi alle metriche calcolate (bar plot e line chart).
- Plot della matrice di confusione calcolata per ogni iterazione.
- Box plot delle metriche (per i metodi Leave One Out e Random Subsampling).
- Salvataggio delle metriche in un apposito file .xlsx (con possibilità di nominare il file generato).
---

## Dataset Utilizzato

Il dataset fornito contiene le seguenti variabili:

### **Anteprima del Dataset**

| Blood Pressure | Mitoses | Sample Code Number | Normal Nucleoli | Single Epithelial Cell Size | Uniformity of Cell Size | Clump Thickness | Heart Rate | Marginal Adhesion | Bland Chromatin | Class Type | Uniformity of Cell Shape | Bare Nucleix |
|----------------|---------|--------------------|-----------------|----------------------------|-------------------------|-----------------|------------|-------------------|-----------------|------------|--------------------------|--------------|
| 95             | 1       | 1000025.0         | 1               | 2.0                        | 1.0                     | 5.0             | 63         | 1.0               | 3.0             | 2.0        | 1.0                      | 1.0          |
| 100            | 1       | 1002945.0         | 2               | 7.0                        | 4.0                     | 5.0             | 66         | 5.0               | 3.0             | 2.0        | 4.0                      | 10.0         |
| 112            | 1       | 1015425.0         | 1               | 2.0                        | NaN                     | NaN             | 72         | 1.0               | 3.0             | NaN        | 1.0                      | 2.0          |
| 99             | 1       | 1016277.0         | 7               | 3.0                        | 8.0                     | 6.0             | 98         | 1.0               | 3.0             | 2.0        | 8.0                      | 4.0          |
| 122            | 1       | 1017023.0         | 1               | 2.0                        | 1.0                     | 4.0             | 66         | 3.0               | 3.0             | 2.0        | 1.0                      | 1.0          |

Nel progetto, sono state utilizzate le seguenti features e label

- **Features (1-10 per ogni caratteristica):**
  - Clump Thickness
  - Uniformity of Cell Size
  - Uniformity of Cell Shape
  - Marginal Adhesion
  - Single Epithelial Cell Size
  - Bare Nuclei
  - Bland Chromatin
  - Normal Nucleoli
  - Mitoses
- **Label di classe:**
  - 2 per **benigno**
  - 4 per **maligno**

## Utilizzo
Per eseguire il codice, basta avviare lo script principale con il comando:

```bash
python main.py
```

Durante l'esecuzione, il programma chiederà all'utente di specificare i seguenti parametri:

1. **Preprocessing dei dati**:
   - Scelta tra **standardizzazione** e **normalizzazione** delle feature.
   - Gestione dei valori NaN (mancanti) scegliendo tra:
     - Sostituzione con **media**, **mediana** o **moda**.
     - Eliminazione delle righe con dati mancanti.

2. **Tipologia di distanza da utilizzare**:
   - **1 - Distanza Euclidea**
   - **2 - Distanza Manhattan**
   - **3 - Distanza Minkowski**
   - **4 - Distanza Chebyshev**
   - **5 - Distanza Coseno**

2. **Metodo di suddivisione del dataset**:
   - **1 - Holdout**: Il dataset viene diviso in training e test set in base alla percentuale specificata (es. 70% train, 30% test).
   - **2 - Leave-One-Out**: Ogni campione viene utilizzato come test set in iterazioni successive (K specificato dall'utente).
   - **3 - Random Subsampling**: Il dataset viene diviso casualmente in training e test set più volte (K specificato dall'utente).


3. **Numero di vicini (*k*)**:
   - L’utente può specificare il valore di *k* (numero di vicini, diverso da K) per il classificatore k-NN.

4. **Scelta delle metriche di valutazione**:
   - L'utente può selezionare una o più metriche tra:
     - **Accuracy Rate (AR)** - Percentuale di predizioni corrette rispetto al totale dei campioni: **AR = (TP + TN) / (TP + TN + FP + FN)**
     - **Error Rate (ER)** - Percentuale di errori nel modello: **ER = (FP + FN) / (TP + TN + FP + FN)**
     - **Sensitivity (True Positive Rate - TPR)** - Probabilità di classificare correttamente un campione positivo: **TPR = TP / (TP + FN)**
     - **Specificity (True Negative Rate - TNR)** - Probabilità di classificare correttamente un campione negativo: **TNR = TN / (TN + FP)**
     - **Geometric Mean (G-Mean)** - Media geometrica di Sensitivity e Specificity, utile nei dataset sbilanciati: **G-Mean = sqrt(TPR × TNR)**
     - **Area Under the Curve (AUC-ROC)** - Misura complessiva delle prestazioni del modello sulla curva ROC: **AUC = ∫ TPR(FPR) dFPR**


5. **Formato dell'output**:
   - Possibilità di esportare i risultati in un file `.xlsx` e visualizzare i grafici generati.

---

### **Come visualizzare ed interpretare i risultati ottenuti**
Dopo l'esecuzione, il programma restituirà:

- **Metriche di valutazione**: i risultati delle metriche selezionate verranno stampati a schermo e salvati in un file `.xlsx`.
- **Grafici di analisi**:
  - **Bar plot** e **line chart** che mostrano l’andamento delle metriche rispetto ai vari test.
  - **Matrice di confusione**, utile per visualizzare il numero di classificazioni corrette e errate.
  - **ROC Curve**, utile per analizzare il bilanciamento tra True Positive Rate (TPR) e False Positive Rate (FPR)
- **File di output (`model_performance.xlsx`)**:
  - Contiene i risultati numerici delle metriche selezionate.
  - Può essere utilizzato per ulteriori analisi o confronti tra esperimenti.

Interpretazione dei risultati:
- **Accuracy alta** → Il modello classifica bene i tumori nel dataset.
- **Error rate elevato** → Il modello commette molti errori, si potrebbe rivedere il valore di *k* o migliorare la gestione dei dati mancanti.
- **Sensitivity e Specificity** → Permettono di capire se il modello ha problemi nel distinguere tumori benigni da maligni.
- **Geometric Mean e AUC**: 
    - **G-Mean** alto indica un buon bilanciamento tra TPR e TNR.
    - **AUC vicino a 1** indica un buon modello, mentre **AUC ≈ 0.5** suggerisce che il modello fa previsioni casuali.

Se serve maggiore personalizzazione nell'output, l'utente può modificare il codice nelle sezioni dedicate alle metriche e alla visualizzazione grafica.


## Docker 
Per eseguire correttamente l'immagine docker su ambiente ospite, è necessario anzitutto scaricare la cartella nel proprio dispositivo.
Se l'utente è in grado di usare una macchina con OS Linux (dove Docker è nativo dell'ambiente), verificare l'installazione di Docker tramite:

''' docker --version '''

se Docker non risulta installato, consultare 'https://docs.docker.com/engine/install/ubuntu/'.
Successivamente, spostarsi nella directory dell'applicazione da eseguire:

''' cd path_personale/Progetto_FIA_24-25_Gruppo2/ '''

Una volta che ci troviamo nella directory, creare l'immagine dell'applicazione:

''' docker build -t my-python-app . '''

A questo punto verrà letto il file 'Dockerfile' nella cartella, e dopo qualche secondo saremo pronti a eseguire l'applicazione eseguendo il comando:

''' docker run -it -v /home/osboxes/Progetto_FIA_24-25_Gruppo2/output:/app/output my-python-app '''.

che permette l'avvio di un container.
In particolar modo, il decordatore '-it' permette l'eseguibilità in maniera interattiva dell'applicazione, '-v' consente invece il salvataggio dell'output dell'applicazione su una cartella su disco remoto.

## Dipendenze (file requirements.txt)
- Python (3.x)
- Numpy
- Pandas
- Random
- Matplotlib

## Autori
- Gruppo 2 del corso di Fondamenti di Intelligenza Artificiale.
