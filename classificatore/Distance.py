import numpy as np
from abc import ABC, abstractmethod

# Definizione dell'interfaccia con un metodo astratto
class DistanzaStrategy(ABC):
    @abstractmethod
    def calcola(self, X1, X2):
        """
        Metodo astratto per calcolare la distanza tra due insiemi di punti.
        """
        pass
    
# Implementazione della distanza euclidea
class DistanzaEuclidea(DistanzaStrategy):
    def calcola(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Calcola la distanza euclidea tra due insiemi di punti.
        
        :param X1: np.ndarray - (n, d) Array di punti.
        :param X2: np.ndarray - (m, d) Array di punti.
        :return: np.ndarray - (n, m) Matrice delle distanze euclidee.
        """
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=2))
    
# Implementazione della distanza di Manhattan
class DistanzaManhattan(DistanzaStrategy):
    def calcola(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Calcola la distanza di Manhattan tra due insiemi di punti.
        
        :param X1: np.ndarray - (n, d) Array di punti.
        :param X2: np.ndarray - (m, d) Array di punti.
        :return: np.ndarray - (n, m) Matrice delle distanze Manhattan.
        """
        diff = np.abs(X1[:, np.newaxis, :] - X2[np.newaxis, :, :])
        return np.sum(diff, axis=2)
    
# Implementazione della distanza di Chebyshev
class DistanzaChebyshev(DistanzaStrategy):
    def calcola(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Calcola la distanza di Chebyshev tra due insiemi di punti.
        
        :param X1: np.ndarray - (n, d) Array di punti.
        :param X2: np.ndarray - (m, d) Array di punti.
        :return: np.ndarray - (n, m) Matrice delle distanze di Chebyshev.
        """
        diff = np.abs(X1[:, np.newaxis, :] - X2[np.newaxis, :, :])
        return np.max(diff, axis=2)

# Implementazione della distanza di Minkowski
class DistanzaMinkowski(DistanzaStrategy):
    def __init__(self, p: int = 3):
        """
        Inizializza la metrica di Minkowski con un parametro p.
        
        :param p: int - Esponente della norma Minkowski.
        """
        self.p = p

    def calcola(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Calcola la distanza di Minkowski tra due insiemi di punti.
        
        :param X1: np.ndarray - (n, d) Array di punti.
        :param X2: np.ndarray - (m, d) Array di punti.
        :return: np.ndarray - (n, m) Matrice delle distanze Minkowski.
        """
        diff = np.abs(X1[:, np.newaxis, :] - X2[np.newaxis, :, :])
        return np.sum(diff**self.p, axis=2)**(1/self.p)

# Implementazione della distanza coseno
class DistanzaCoseno(DistanzaStrategy):
    def calcola(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Calcola la distanza coseno tra due insiemi di punti.
        
        :param X1: np.ndarray - (n, d) Array di punti.
        :param X2: np.ndarray - (m, d) Array di punti.
        :return: np.ndarray - (n, m) Matrice delle distanze coseno.
        """
        X1_norm = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
        X2_norm = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
        similarity = np.dot(X1_norm, X2_norm.T)
        return 1 - similarity  # Distanza coseno = 1 - Similarità coseno

    
class DistanceFactory:
    '''
    Factory per la creazione di tipi di distanze diverse da utilizzare nel classificatore
    :param choice_distance: stringa che seleziona il tipo di distanza
    :return float: distanza
    '''
    def create(self, choice_distance: int) -> float:
        if choice_distance == 1:
            return DistanzaEuclidea()
        elif choice_distance == 2:
            return DistanzaManhattan()
        elif choice_distance == 3:
            return DistanzaMinkowski()
        elif choice_distance == 4:
            return DistanzaChebyshev()
        elif choice_distance == 5:
            return DistanzaCoseno()
        else:
            raise ValueError("Tipo di distanza non supportato")