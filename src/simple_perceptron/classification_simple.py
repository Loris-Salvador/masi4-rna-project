import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class ClassificationSimplePerceptron():
    
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
        return 1 if z >= 0 else -1

    def train(self, learning_rate, epochs):

        for epoch in range(epochs):

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

        for i in range(len(self.entries)):
            z = np.dot(self.entries[i], self.weights)
            y_pred = self.__activation(z)

    def plot_decision_boundary(self):
        # Points de données (sans la colonne de biais)
        X = self.entries[:, 1:]
        y = self.exp_outputs

        # Tracer les points (avec différentes couleurs selon la classe)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Classe 1')
        plt.scatter(X[y == -1, 0], X[y == -1, 1], color='blue', label='Classe -1')

        # Calcul de la droite de séparation : w_0 + w_1 * x1 + w_2 * x2 = 0
        # Résoudre pour x2 : x2 = (-w_0 - w_1 * x1) / w_2
        x1_values = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        x2_values = (-self.weights[0] - self.weights[1] * x1_values) / self.weights[2]

        # Tracer la droite de séparation
        plt.plot(x1_values, x2_values, color='black', label='Frontière de décision')

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Perceptron - Frontière de Décision')
        plt.legend()
        plt.show()