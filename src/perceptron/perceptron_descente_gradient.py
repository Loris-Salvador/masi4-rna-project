from perceptron.perceptron_base import PerceptronBase



class PerceptronDescenteGradient(PerceptronBase):

    def __init__(self, data_path, graph_name="Descente Gradient"):
        super().__init__(data_path=data_path, graph_name=graph_name)

    def train(self, learning_rate, epochs):
        for epoch in range(epochs):
            z = self.entries.dot(self.weights)
            error = self.exp_outputs - z
            grad  = self.entries.T.dot(error)
            self.weights += learning_rate * grad


    def display_graph_classification(self):
        super().display_graph_classification()


    def display_graph_regression(self):
        super().display_graph_regression