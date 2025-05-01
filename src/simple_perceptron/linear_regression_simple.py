import numpy as np
import matplotlib.pyplot as plt
from simple_perceptron.simple_perceptron_base import SimplePerceptronBase


class LinearRegressionSimplePerceptron(SimplePerceptronBase):
    def __init__(self, data_path):
        super().__init__(data_path = data_path)

    def train(self, learning_rate, epochs):
        for epoch in range(epochs):
            for i in range(len(self.entries)):
                x_i = self.entries[i]
                y_true = self.exp_outputs[i]

                y_pred = np.dot(x_i, self.weights)
                error = y_true - y_pred

                self.weights += learning_rate * error * x_i


    def plot(self):
        plt.scatter(self.entries[:, 1], self.exp_outputs, color='blue', label='Données réelles')
        x_range = np.linspace(min(self.entries[:, 1]), max(self.entries[:, 1]), 100)
        y_range = [self.predict([x]) for x in x_range]
        plt.plot(x_range, y_range, color='red', label='Régression linéaire')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

    def predict(self, x):
        x = np.insert(x, 0, 1)  # ajouter le biais
        return np.dot(x, self.weights)