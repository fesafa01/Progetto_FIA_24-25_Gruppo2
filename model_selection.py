# Funzione per la scelta del modello di divisione del dataset in train e test set
class ModelSelection:

    #La funzione di selezione non ha nulla in ingresso e restituisce la scelta dell'utente sul metodo da applicare
    @staticmethod
    def model_selection():
        print("Scegli il metodo di divisione del dataset:")
        print("1 - Holdout")
        print("2 - Leave One Out")
        print("3 - Random Subsampling")
        choice = input("Inserisci il numero della tua scelta: ")

        # Dizionario per mappare la scelta numerica al relativo metodo
        methods = {
            '1': 'Holdout',
            '2': 'Leave One Out',
            '3': 'Random Subsampling'
        }
        
        chosen_method = methods.get(choice, 'Scelta non valida') # Recupera il metodo corrispondente, se non esiste restituisce 'Scelta non valida'
        print(f"Utente ha scelto: {chosen_method}") # Stampa del metodo scelto

        return choice
    



        