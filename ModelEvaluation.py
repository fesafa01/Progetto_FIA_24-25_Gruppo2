from abc import ABC, abstractmethod
from Holdout import Holdout 
from LeaveOneOut import LeaveOneOut
from RandomSubsampling import RandomSubsampling
#from appoggio import metrics_calculator

#Implementazione del metodo di validazione Holdout, che divide il dataset in due parti: training e test set
# in particolare, il test set è una porzione del dataset che non verrà utilizzata per l'addestramento del modello
# ma solo per la sua valutazione

class ModelSelection(ABC):
    '''
    Classe astratta per la selezione del modello di divisione del dataset in training e test set.
    '''
    def __init__(self, test_size=None, K=None, num_splits=None):
        """
        Costruttore della classe base per tutte le strategie di validazione.

        :param test_size: float, dimensione del test set (se applicabile).
        :param K: int, numero di iterazioni per Leave-One-Out (se applicabile).
        :param num_splits: int, numero di suddivisioni per Random Subsampling.
        """
        self.test_size = test_size
        self.K = K
        self.num_splits = num_splits

    @abstractmethod
    def splitter(self, X, y):
        """Metodo astratto per dividere il dataset in training e test set."""
        pass

    @abstractmethod
    def run(self, X, y, k=3):
        """Metodo astratto per eseguire la validazione e testare il modello."""
        pass

    @abstractmethod
    def evaluate(self, X, y, k=3):
        """Metodo astratto per calcolare le metriche di valutazione."""
        pass


class ModelEvaluationFactory:
    @staticmethod
    def get_validation_strategy(choice, X):
        """
        Factory per restituire l'istanza della strategia di validazione scelta.

        :param method: string, nome della strategia ('holdout', 'leave_one_out', 'random_subsampling')

        :return: Istanza della classe di validazione corrispondente, in base alla scelta dell'utente.
        """

        if choice == "1":
            test_size = float(input("Inserisci il valore percentuale che rappresenta la dimensione del test set rispetto all'intero set di dati: "))
            return Holdout(test_size)
        elif choice == "2":
            K = int(input(f"Inserisci il numero di esperimenti (intero) tra 1 e {len(X)}: "))
            return LeaveOneOut(K, X)  
        elif choice == "3":
            test_size = float(input("Inserisci il valore percentuale che rappresenta la dimensione del test set rispetto all'intero set di dati: "))
            num_splits = int(input("Inserisci il numero di splits da realizzare nel metodo: "))
            return RandomSubsampling(test_size, num_splits)
        else:
            raise ValueError(f"Metodo di validazione non riconosciuto.")