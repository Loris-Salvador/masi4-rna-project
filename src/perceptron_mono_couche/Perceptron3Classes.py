import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron3Classes:
    def __init__(self, data_path):
        # Charger les données depuis le fichier CSV
        data = pd.read_csv(data_path, header=None).values

        # Les deux premières colonnes comme caractéristiques
        self.X = data[:, :2]
        # Les trois dernières colonnes comme classes (one-hot)
        self.y = data[:, 2:]

        # Ajouter la colonne de biais (1) à X
        self.X = np.c_[np.ones((self.X.shape[0], 1)), self.X]

        # Initialisation aléatoire des poids
        self.weights = np.random.randn(self.y.shape[1], self.X.shape[1])

    def activation(self, z):
        # Activation de type signe : +1 ou -1
        return 1 if z >= 0 else -1

    def train(self, learning_rate, epochs):
        # Entraînement du modèle
        for epoch in range(epochs):
            for i in range(len(self.X)):
                x_i = self.X[i]
                y_true = self.y[i]

                # Parcours des classes
                for class_idx in range(3):  # On sait que c'est exactement 3 classes
                    z = np.dot(self.weights[class_idx], x_i)
                    y_pred = self.activation(z)

                    # Erreur pour cette classe
                    error = y_true[class_idx] - y_pred
                    # Mise à jour des poids
                    self.weights[class_idx] += learning_rate * error * x_i

    def predict(self, x):
        # Prédire la classe d'un nouvel exemple
        x = np.insert(x, 0, 1)  # Ajouter la valeur 1 pour le biais
        scores = [np.dot(w, x) for w in self.weights]
        return np.argmax(scores)  # Retourne l'indice de la classe avec le score maximal

    def plot(self, title="Perceptron 3 classes"):
        # Vérifier que les données sont en 2D pour pouvoir les afficher
        if self.X.shape[1] != 3:
            print("Les données ne sont pas en 2D, impossible d'afficher le graphique.")
            return

        # Choisir des couleurs pour chaque classe
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(3)]

        # Séparer les données par classe
        labels = np.argmax(self.y, axis=1)
        for i in range(3):  # On sait que c'est 3 classes
            plt.scatter(self.X[labels == i][:, 1], self.X[labels == i][:, 2], c=[colors[i]], label=f"Classe {i}")

        # Définir la plage des axes pour le fond de la grille
        x_min, x_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        y_min, y_max = self.X[:, 2].min() - 0.5, self.X[:, 2].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid = np.c_[np.ones((xx.size, 1)), xx.ravel(), yy.ravel()]
        preds = [self.predict(point[1:]) for point in grid]
        preds = np.array(preds).reshape(xx.shape)

        # Afficher les frontières de décision
        plt.contourf(xx, yy, preds, alpha=0.2, levels=[-0.5, 0.5, 1.5, 2.5], colors=colors)

        # Labels et titre
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title(title)
        plt.legend()
        plt.show()
