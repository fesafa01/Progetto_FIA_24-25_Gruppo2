# Progetto_FIA_24-25_Gruppo2

## Descrizione
Spazio di lavoro per lo sviluppo del progetto del corso di Fondamenti di Intelligenza Artificiale dell'Università Campus Bio-Medico di Roma. Il progetto include lo sviluppo di un classificatore k-NN implementato da zero.

## Funzionalità
- Selezione interattiva nella scelta del modello da applicare per dividere il dataset ed eventuali parametri caratteristici di questi modelli.
- Holdout: metodo che permette la suddivisione di features e labels in train e test usati per l'addestramento e la valutazione del modello.
- Leave-one-out: metodo che permette di suddivere per tante volte quanto è il numero di campioni, il dataset in training e test set, costituito in ogni iterazione da un solo campione. Il metodo permette di addestrare e testare il classificatore al fine di validarlo.
- Random Subsampling: metodo che suddivide il dataset in training e test set più volte in modo casuale per ottenere una valutazione più stabile delle prestazioni del modello.
- Classificatore k-NN: Un modello che predice le etichette dei dati basandosi sui k nearest neighbours, utilizzando la distanza euclidea (modificabile se necessario).
- Calcolo delle seguenti metriche per ogni metodo di validazione del modello: accuracy rate (AR), error rate (ER), sensitivity (TPR), specificity (TNR), geometric mean, area under the curve. Vengono stampate solo le metriche richieste dall'utente.

## Utilizzo
Per utilizzare il classificatore k-NN, seguire questi passaggi:
1. Clonare il repository.
2. Assicurarsi di avere Python installato con le librerie `numpy` e `random`.
3. Eseguire il file `main.py` per addestrare e testare il modello k-NN.

## Dipendenze
- Python (3.x)
- Numpy
- Pandas
- Random

## Autori
- Gruppo 2 del corso di Fondamenti di Intelligenza Artificiale