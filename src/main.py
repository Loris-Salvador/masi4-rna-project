from perceptron_mono_couche.Perceptron3Classes import Perceptron3Classes
from perceptron_mono_couche.Perceptron4Classes import Perceptron4Classes
from simple_perceptron.simple_perceptron import SimplePerceptron
from perceptron_gradient.perceptron_descente_gradient import PerceptronDescenteGradient

from data_path import TABLE_AND, TABLE_2_9, TABLE_2_10, TABLE_2_11, TABLE_3_1, TABLE_3_5



def main():

    descente_gradient()

    #simple_perceptron()

    #monocouche()


def simple_perceptron():
    model_and = SimplePerceptron(data_path=TABLE_AND, graph_name="Perceptron Simple Classification AND")

    model_and.train(learning_rate=0.1, epochs=500)

    model_and.display_graph_classification()

    model_2_9 = SimplePerceptron(data_path=TABLE_2_9, graph_name="Perceptron Simple Classification linéairment séparable")

    model_2_9.train(0.1, 500)

    model_2_9.display_graph_classification()

    model_2_10 = SimplePerceptron(data_path=TABLE_2_10, graph_name="Perceptron Simple Classification non linéairment séparable")

    model_2_10.train(0.1, 500)

    model_2_10.display_graph_classification()

    model_2_11 = SimplePerceptron(data_path=TABLE_2_11, graph_name="Perceptron Simple Regression")

    model_2_11.train(0.1, 500)

    model_2_11.display_graph_regression()


def descente_gradient():
    model_and = PerceptronDescenteGradient(data_path=TABLE_AND, graph_name="Perceptron Gradient Classification AND")

    model_and.train(learning_rate=0.1, epochs=500)

    model_and.display_graph_classification()

    model_2_9 = PerceptronDescenteGradient(data_path=TABLE_2_9, graph_name="Perceptron Gradient Classification linéairment séparable")

    model_2_9.train(0.001, 500)

    model_2_9.display_graph_classification()

    model_2_10 = PerceptronDescenteGradient(data_path=TABLE_2_10, graph_name="Perceptron Gradient Classification non linéairment séparable")

    model_2_10.train(0.001, 500)

    model_2_10.display_graph_classification()

    model_2_11 = PerceptronDescenteGradient(data_path=TABLE_2_11, graph_name="Perceptron Gradient Regression")

    model_2_11.train(learning_rate=1e-5, epochs=2000)

    model_2_11.display_graph_regression()


def monocouche():
    print("Choisissez le nombre de classe pour le perceptron monocouche")
    print("1. 3 classes")
    print("2. 4 classes")
    table_choice = input("Entrez le numéro de la table (1 ou 2) : ").strip()
    if table_choice == "1":
        data_path = TABLE_3_1
        model = Perceptron3Classes(data_path=data_path)
    if table_choice == "2":
        data_path = TABLE_3_5
        model = Perceptron4Classes(data_path=data_path)


    learning_rate = float(input("Entrez le taux d'apprentissage (par exemple 0.1) : "))
    epochs = int(input("Entrez le nombre d'époques (par exemple 500) : "))

    model.train(learning_rate=learning_rate, epochs=epochs)

    if table_choice == '2':
        model.evaluate()
    else:
        model.plot()



if __name__ == "__main__":
    main()

