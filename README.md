# Progetto_FIA_24-25_Gruppo2

## Descrizione
Spazio di lavoro per lo sviluppo del progetto del corso di Fondamenti di Intelligenza Artificiale dell'Università Campus Bio-Medico di Roma. Il progetto include lo sviluppo di un classificatore k-NN implementato da zero.

## Funzionalità
- Inserimento dinamico del percorso e del nome del file, contenente il dataset.
- Selezione interattiva nella scelta del modello da applicare per dividere il dataset ed eventuali parametri caratteristici di questi modelli.
- Holdout: metodo che permette la suddivisione di features e labels in train e test usati per l'addestramento e la valutazione del modello.
- Leave-one-out: metodo che permette di suddivere per tante volte quanto è il numero di campioni, il dataset in training e test set, costituito in ogni iterazione da un solo campione. Il metodo permette di addestrare e testare il classificatore al fine di validarlo.
- Random Subsampling: metodo che suddivide il dataset in training e test set più volte in modo casuale per ottenere una valutazione più stabile delle prestazioni del modello.
- Classificatore k-NN: Un modello che predice le etichette dei dati basandosi sui k nearest neighbours, utilizzando la distanza euclidea (modificabile se necessario).
- Calcolo delle seguenti metriche per ogni metodo di validazione del modello: accuracy rate (AR), error rate (ER), sensitivity (TPR), specificity (TNR), geometric mean, area under the curve. Vengono stampate solo le metriche richieste dall'utente.
- Plot dei grafici relativi alle metriche calcolate (bar plot e line chart).
- Plot della matrice di confusione calcolata.
- Salvataggio delle metriche in un apposito file .xlsx (con possibilità di nominare il file generato).

## Utilizzo
Per utilizzare il classificatore k-NN, seguire questi passaggi:
1. Clonare il repository.
2. Assicurarsi di avere Python installato con le librerie `numpy` e `random`.
3. Eseguire il file `main.py` per addestrare e testare il modello k-NN.

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

''' docker run -it -v path_personale/Progetto_FIA_24-25_Gruppo2:/output my-python-app '''.

che permette l'avvio di un container.
In particolar modo, il decordatore '-it' permette l'eseguibilità in maniera interattiva dell'applicazione, '-v' consente invece il salvataggio dell'output dell'applicazione su una cartella su disco remoto.

## Dipendenze (file requirements.txt)
- Python (3.x)
- Numpy
- Pandas
- Random
- Matplotlib

## Autori
- Gruppo 2 del corso di Fondamenti di Intelligenza Artificiale