from perceptron.perceptron_base import PerceptronBase
import numpy as np


class PerceptronAdaline(PerceptronBase):

    def __init__(self, data_path, graph_name):
        super().__init__(data_path=data_path, graph_name=graph_name)

    def train(self, learning_rate, epochs, tolerance=1e-3):
        for epoch in range(epochs):
            total_error = 0.0

            for i in range(len(self.entries)):
                entry = self.entries[i]
                expected_output = self.expected_outputs[i]

                prediction = np.dot(self.weights, entry)
                error = expected_output - prediction
                total_error += error ** 2

                for j in range (len(self.weights)):
                    self.weights[j] = self.weights[j] + learning_rate * error * entry[j]

            E_moy = total_error / len(self.entries)

            if E_moy < tolerance:
                break

        print(f"Apprentissage terminé : le nombre d'epoch maximum a été atteint ({epochs})")

    def display_graph_classification(self):
        super().display_graph_classification()


    def display_graph_regression(self):
        super().display_graph_regression()