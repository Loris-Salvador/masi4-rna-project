import numpy as np
from perceptron.perceptron_base import PerceptronBase

class SimplePerceptron(PerceptronBase):
    
    def __init__(self, data_path, graph_name):
        super().__init__(data_path=data_path, graph_name=graph_name)

    def __activation(self, z):
        return 1 if z >= 0 else -1

    def train(self, learning_rate, epochs):

        for epoch in range(epochs):

            for i in range(len(self.entries)):
                x_i = self.entries[i]
                y_true = self.exp_outputs[i]

                z = np.dot(x_i, self.weights)
                y_pred = self.__activation(z)

                error = y_true - y_pred

                self.weights += learning_rate * error * x_i


    def display_graph_classification(self):
        super().display_graph_classification()


    def display_graph_regression(self):
        super().display_graph_regression