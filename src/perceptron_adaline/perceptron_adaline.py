import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class PerceptronAdaline:
    """
    Perceptron ADALINE (Widrow–Hoff) :
    apprentissage stochastique sur l'erreur quadratique.
    """
    def __init__(self, data_path, graph_name="ADALINE"):
        self.graph_name = graph_name
        data = pd.read_csv(data_path, header=None)
        X = data.iloc[:, :-1].values
        self.exp_outputs = data.iloc[:,  -1].values
        # biais
        self.entries = np.c_[np.ones((X.shape[0], 1)), X]
        # poids init
        self.weights = np.random.randn(self.entries.shape[1])

    def train(self, learning_rate, epochs):
        """
        À chaque exemple (stochastique) :
          z = w·x
          error = d - z    # erreur continue
          w += lr * error * x
        """
        for epoch in range(epochs):
            for x_i, d_i in zip(self.entries, self.exp_outputs):
                z     = x_i.dot(self.weights)
                error = d_i - z
                self.weights += learning_rate * error * x_i

                

    def display_graph_classification(self):
        X = self.entries[:, 1:]
        y = self.exp_outputs

        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Classe 1')
        plt.scatter(X[y == -1, 0], X[y == -1, 1], color='blue', label='Classe -1')


        x1_values = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        x2_values = (-self.weights[0] - self.weights[1] * x1_values) / self.weights[2]

        plt.plot(x1_values, x2_values, color='black', label='Frontière de décision')

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title(self.graph_name)
        plt.legend()
        plt.show()



    def display_graph_regression(self):
        X = self.entries[:, 1:]  # On enlève le biais pour tracer
        y = self.exp_outputs

        if X.shape[1] != 1:
            print("Impossible d'afficher une régression en 2D avec plus d'une variable explicative.")
            return

        plt.scatter(X, y, color='blue', label='Données réelles')

        x_range = np.linspace(np.min(X), np.max(X), 100)
        x_range_with_bias = np.c_[np.ones((100, 1)), x_range]
        y_pred = x_range_with_bias.dot(self.weights)

        plt.plot(x_range, y_pred, color='red', label='Régression par perceptron')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(self.graph_name)
        plt.legend()
        plt.show()
