import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Perceptron4Classes:
    def __init__(self, data_path):
        data = pd.read_csv(data_path, header=None).values

        # Nous avons 25 features et 4 classes
        input_size = 25
        output_size = 4

        # Séparation des caractéristiques (X) et des labels (y)
        self.X = data[:, :input_size]
        self.y = data[:, input_size:]

        # Ajouter la colonne du biais (x0 = 1)
        self.X = np.c_[np.ones((self.X.shape[0], 1)), self.X]

        # Initialisation des poids avec des valeurs aléatoires
        self.weights = np.random.randn(output_size, self.X.shape[1])

    def activation(self, z):
        # Fonction d'activation signum (fonction seuil)
        return 1 if z >= 0 else -1

    def train(self, learning_rate, epochs):
        # Entraînement du perceptron sur plusieurs époques
        for epoch in range(epochs):
            for i in range(len(self.X)):
                x_i = self.X[i]
                y_true = self.y[i]

                # Calcul des prédictions et mise à jour des poids
                for class_idx in range(self.y.shape[1]):
                    z = np.dot(self.weights[class_idx], x_i)
                    y_pred = self.activation(z)

                    # Calcul de l'erreur
                    error = y_true[class_idx] - y_pred

                    # Mise à jour des poids pour chaque classe
                    self.weights[class_idx] += learning_rate * error * x_i

    def predict(self, x):
        # Prédiction pour une nouvelle entrée x
        x = np.insert(x, 0, 1)  # Ajouter le biais (x0 = 1)
        scores = [np.dot(w, x) for w in self.weights]
        return np.argmax(scores)  # Retourne la classe avec le score maximal

    def evaluate(self):
        print("Évaluation des prédictions :\n")
        correct = 0
        for i in range(len(self.X)):
            x_i = self.X[i][1:] 
            true_class = np.argmax(self.y[i])
            predicted_class = self.predict(x_i)
            status = "✔️ Correct" if predicted_class == true_class else "❌ Faux"
            print(f"Exemple {i + 1}: Attendu = Classe {true_class}, Prédit = Classe {predicted_class} -> {status}")
            if predicted_class == true_class:
                correct += 1
        accuracy = correct / len(self.X) * 100
        print(f"\nTaux de réussite : {accuracy:.2f}%")

