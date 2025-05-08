import numpy as np
from perceptron.perceptron_base import PerceptronBase


def activation(z):
    return 1 if z >= 0 else -1


class PerceptronSimple(PerceptronBase):
    
    def __init__(self, data_path, graph_name):
        super().__init__(data_path=data_path, graph_name=graph_name)

    def train(self, learning_rate, epochs):
        for epoch in range(epochs):
            nb_error = 0

            for i in range(len(self.entries)):
                entry = self.entries[i]
                expected_output = self.expected_outputs[i]

                scalaire_product = np.dot(entry, self.weights)
                prediction = activation(scalaire_product)

                error = expected_output - prediction

                if error != 0:
                    nb_error += 1
                    
                    
                for j in range (len(self.weights)):
                    self.weights[j] = self.weights[j] + learning_rate * error * entry[j]

            if nb_error == 0:
                print(f"Apprentissage terminé à l'époque {epoch} : le nombre d'erreurs a atteint 0")
                break

        print(f"Apprentissage terminé : le nombre d'epoch maximum a été atteint ({epochs})")

    def display_graph_classification(self):
        super().display_graph_classification()


    def display_graph_regression(self):
        super().display_graph_regression()