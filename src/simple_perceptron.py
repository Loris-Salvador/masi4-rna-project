import numpy as np
import pandas as pd


class SimplePerceptron:
    
    def __init__(self, data_path):
        data = pd.read_csv(data_path, header=None)

        #toutes les lignes et toutes les colonnes sauf derniere
        self.entries = data.iloc[:, :-1].values

        #toutes les lignes et uniquement la derniere colonne
        self.exp_outputs = data.iloc[:, -1].values 

        # Ajouter une colonne de 1 pour le biais à self.entries
        self.entries = np.c_[np.ones((self.entries.shape[0], 1)), self.entries]

        #shape[0] donne le nb lignes et shape[1] le nb colonnes
        n_features = self.entries.shape[1]

        self.weights = np.random.randn(n_features)


    def __activation(self, z):
        return 1 if z >= 0 else 0


    def train(self, learning_rate, epochs):

        for epoch in range(epochs):
            print(f"Époque {epoch+1}")
            for i in range(len(self.entries)):
                x_i = self.entries[i]
                y_true = self.exp_outputs[i]

                # Produit scalaire + biais
                z = np.dot(x_i, self.weights)
                y_pred = self.__activation(z)

                # Erreur
                error = y_true - y_pred

                # Mise à jour des poids et du biais
                self.weights += learning_rate * error * x_i

                print(f"  Entrée: {x_i}, Sortie attendue: {y_true}, Prédite: {y_pred}, Erreur: {error}")
            print()

        print("\nTest du perceptron après entraînement :")
        for i in range(len(self.entries)):
            z = np.dot(self.entries[i], self.weights)
            y_pred = self.__activation(z)
            print(f"Entrée: {self.entries[i]} → Prédit: {y_pred}, Attendu: {self.exp_outputs[i]}")




def main():
    sp = SimplePerceptron(data_path="./data/table_AND.csv")

    sp.train(learning_rate=0.1, epochs=30)


if __name__ == "__main__":
    main()