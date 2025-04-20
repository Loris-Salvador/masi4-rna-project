import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegressionSimplePerceptron:
    def __init__(self, data_path):
        data = pd.read_csv(data_path, header=None)

        self.entries = data.iloc[:, 0].values.reshape(-1, 1)
        self.targets = data.iloc[:, 1].values

        self.entries = np.c_[np.ones((self.entries.shape[0], 1)), self.entries]

        self.weights = np.random.randn(self.entries.shape[1])

    def train(self, learning_rate, epochs):
        for epoch in range(epochs):
            #total_loss = 0
            for i in range(len(self.entries)):
                x_i = self.entries[i]
                y_true = self.targets[i]

                y_pred = np.dot(x_i, self.weights)
                error = y_true - y_pred
                #total_loss += error ** 2

                self.weights += learning_rate * error * x_i


    def plot(self):
        plt.scatter(self.entries[:, 1], self.targets, color='blue', label='Données réelles')
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