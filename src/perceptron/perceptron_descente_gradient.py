from perceptron.perceptron_base import PerceptronBase
import numpy as np


class PerceptronDescenteGradient(PerceptronBase):

    def __init__(self, data_path, graph_name):
        super().__init__(data_path=data_path, graph_name=graph_name)

    def train(self, learning_rate, epochs, threshold=1e-3):
        average_error = 0.0
        for epoch in range(epochs):
            total_error = 0.0
            delta_w_i = np.zeros_like(self.weights)

            for i in range(len(self.entries)):
                entry = self.entries[i]
                expected_output = self.expected_outputs[i]

                prediction = np.dot(self.weights, entry)
                error = expected_output - prediction
                total_error += 0.5 * (error ** 2)

                for j in range (len(delta_w_i)):
                    delta_w_i[j] = delta_w_i[j] + learning_rate * error * entry[j]

            self.weights += delta_w_i

            average_error = total_error / len(self.entries)

            if average_error < threshold:
                print(f"Apprentissage terminé à l'époque {epoch} : le nombre d'erreurs a atteint 0")
                break

        if average_error >= threshold:
            print(f"Apprentissage terminé : le nombre d'epoch maximum a été atteint ({epochs})")

    def display_graph_classification(self):
        super().display_graph_classification()


    def display_graph_regression(self):
        super().display_graph_regression()