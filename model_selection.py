# Funzione per la scelta del modello di divisione del dataset in train e test set
class ModelSelection:

    #La funzione di selezione non ha nulla in ingresso e restituisce la scelta dell'utente sul metodo da applicare
    @staticmethod
    def model_selection():
        print("Scegli il metodo di divisione del dataset:")
        print("1 - Holdout")
        print("2 - Leave One Out")
        print("3 - Metodo 3")
        choice = input("Inserisci il numero della tua scelta: ")
        print(f"Utente ha scelto: {choice}")
        return choice