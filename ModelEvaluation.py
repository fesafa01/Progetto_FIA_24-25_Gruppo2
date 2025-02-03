from abc import ABC, abstractmethod
from Holdout import Holdout 
from LeaveOneOut import LeaveOneOut
from RandomSubsampling import RandomSubsampling

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
    """
    Classe Factory per la creazione delle istanze delle diverse strategie di validazione.

    Questa classe fornisce un metodo statico che permette di istanziare la strategia 
    di validazione scelta dall'utente. Supporta tecniche di validazione come Holdout, 
    Leave-One-Out e Random Subsampling.

    Utilizza parametri opzionali (**kwargs) per rendere flessibile il passaggio 
    delle informazioni necessarie a ciascuna strategia.
    """
        
    @staticmethod
    def get_validation_strategy(choice, **kwargs):
        '''
        Restituisce un'istanza della strategia di validazione selezionata.

        :param choice: str, scelta dell'utente per la strategia di validazione 
                        ("1" per Holdout, "2" per Leave-One-Out, "3" per Random Subsampling).
        :param kwargs: dict, contiene i parametri necessari per la strategia scelta.

        :return: Istanza della classe corrispondente alla scelta dell'utente.

        :raises ValueError: Se la scelta non è valida.
        '''
        if choice == "1":
            return Holdout(kwargs.get("test_size"))
        elif choice == "2":
            return LeaveOneOut(kwargs.get("K"))
        elif choice == "3":
            return RandomSubsampling(kwargs.get("test_size"), kwargs.get("num_splits"))
        else:
            raise ValueError("Metodo di validazione non riconosciuto.")

'''
class ModelEvaluationFactory:
    @staticmethod
    def get_validation_strategy(choice, X):
        """
        Factory per restituire l'istanza della strategia di validazione scelta.

        :param choice: int, valore intero che rappresenta il metodo di validazione scelto dall'utente ('holdout', 'leave_one_out', 'random_subsampling')

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
'''