from perceptron.perceptron_base import PerceptronBase
import numpy as np


class PerceptronDescenteGradient(PerceptronBase):

    def __init__(self, data_path, graph_name):
        super().__init__(data_path=data_path, graph_name=graph_name)

    def train(self, learning_rate, epochs, tolerance=1e-3):
        for epoch in range(epochs):
            total_error = 0.0
            delta_w = np.zeros_like(self.weights)

            for i in range(len(self.entries)):
                x_i = self.entries[i]
                y_true = self.exp_outputs[i]

                y_pred = np.dot(self.weights, x_i)
                error = y_true - y_pred
                total_error += error ** 2

                delta_w += learning_rate * error * x_i

            self.weights += delta_w

            E_moy = total_error / len(self.entries)

            if E_moy < tolerance:
                break

    def display_graph_classification(self):
        super().display_graph_classification()


    def display_graph_regression(self):
        super().display_graph_regression()