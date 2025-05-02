from perceptron.perceptron_base import PerceptronBase


class PerceptronAdaline(PerceptronBase):

    def __init__(self, data_path, graph_name="ADALINE"):
        super().__init__(data_path=data_path, graph_name=graph_name)

    def train(self, learning_rate, epochs):
        for epoch in range(epochs):
            for x_i, d_i in zip(self.entries, self.exp_outputs):
                z = x_i.dot(self.weights)
                error = d_i - z
                self.weights += learning_rate * error * x_i

    def display_graph_classification(self):
        super().display_graph_classification()


    def display_graph_regression(self):
        super().display_graph_regression()