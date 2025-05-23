import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PerceptronBase:
    
    def __init__(self, data_path, graph_name, ecart_type):
        self.graph_name = graph_name

        data = pd.read_csv(data_path, header=None)

        #toutes les lignes et toutes les colonnes sauf dernière
        self.entries = data.iloc[:, :-1].values

        #toutes les lignes et uniquement la dernière colonne
        self.expected_outputs = data.iloc[:, -1].values 

        # Ajouter une colonne de 1 pour le biais
        # shape[0] = ligne et shape[1] = colonne
        self.entries = np.c_[np.ones((self.entries.shape[0], 1)), self.entries]

        self.weights = np.random.normal(0, ecart_type, self.entries.shape[1])   


    def display_graph_classification(self):
        x = self.entries[:, 1:]
        y = self.expected_outputs

        plt.scatter(x[y == 1, 0], x[y == 1, 1], color='red', label='Classe 1')
        plt.scatter(x[y == -1, 0], x[y == -1, 1], color='blue', label='Classe -1')


        x1_values = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)

        #w0 + w1 * x1 + w2 * x2 = 0
        x2_values = (-self.weights[0] - self.weights[1] * x1_values) / self.weights[2]

        plt.plot(x1_values, x2_values, color='black', label='Frontière de décision')

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title(self.graph_name)
        plt.legend()
        plt.show()



    def display_graph_regression(self):
        x = self.entries[:, 1:]
        y = self.expected_outputs

        plt.scatter(x, y, color='blue', label='Données réelles')

        x_range = np.linspace(np.min(x), np.max(x), 100)
        x_range_with_bias = np.c_[np.ones((100, 1)), x_range]
        y_pred = x_range_with_bias.dot(self.weights)

        plt.plot(x_range, y_pred, color='red', label='Régression par perceptron')
        plt.xlabel('x')
        plt.ylabel('Y')
        plt.title(self.graph_name)
        plt.legend()
        plt.show()
