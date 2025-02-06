# Funzione per la scelta del tipo di distanza da utilizzare nel classificatore
class DistanceSelection:

    #La funzione di selezione non ha nulla in ingresso e restituisce la scelta dell'utente sulla distanza da applicare
    @staticmethod
    def distance_selection():
        print("Scegli il tipo di distanza da utilizzare nel classificatore:")
        print("1 - Distanza Euclidea")
        print("2 - Distanza Manhattan")
        print("3 - Distanza Minkowski")
        print("4 - Distanza Chebyshev")
        print("5 - Distanza Coseno")
        choice = input("Inserisci il numero della tua scelta: ")

        # Dizionario per mappare la scelta numerica al relativo metodo
        methods = {
            '1': 'Distanza Euclidea',
            '2': 'Distanza Manhattan',
            '3': 'Distanza Minkowski',
            '4': 'Distanza Chebyshev',
            '5': 'Distanza Coseno'
        }
        
        chosen_method = methods.get(choice, 'Scelta non valida') # Recupera il tipo corrispondente, se non esiste restituisce 'Scelta non valida'
        print(f"Utente ha scelto: \n                    {chosen_method}\n") # Stampa del tipo scelto

        return choice
    