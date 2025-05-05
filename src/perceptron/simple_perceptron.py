import numpy as np
from perceptron.perceptron_base import PerceptronBase


def activation(z):
    return 1 if z >= 0 else -1


class SimplePerceptron(PerceptronBase):
    
    def __init__(self, data_path, graph_name):
        super().__init__(data_path=data_path, graph_name=graph_name)

    def train(self, learning_rate, epochs):
        for epoch in range(epochs):
            nb_error = 0

            for i in range(len(self.entries)):
                x_i = self.entries[i]
                y_true = self.exp_outputs[i]

                z = np.dot(x_i, self.weights)
                y_pred = activation(z)

                error = y_true - y_pred

                if error != 0:
                    nb_error += 1

                self.weights += learning_rate * error * x_i

            if nb_error == 0:
                break


    def display_graph_classification(self):
        super().display_graph_classification()


    def display_graph_regression(self):
        super().display_graph_regression()