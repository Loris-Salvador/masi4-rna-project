from perceptron_mono_couche.Perceptron3Classes import Perceptron3Classes
from perceptron_mono_couche.Perceptron4Classes import Perceptron4Classes
from perceptron.simple_perceptron import   SimplePerceptron
from perceptron.perceptron_descente_gradient import PerceptronDescenteGradient
from perceptron.perceptron_adaline import PerceptronAdaline
from perceptron_multi_couche.MLPModulaire import MLPModulaire
from langage_des_signes.data_loader import LangageSignesData
from langage_des_signes.neural_network import NeuralNetworkScratch

from data_path import TABLE_AND, TABLE_2_9, TABLE_2_10, TABLE_2_11, TABLE_3_1, TABLE_3_5, TABLE_XOR, TABLE_4_12, \
    TABLE_4_14, TABLE_4_17 , TABLE_SIGN_LANGUAGE

def main():

    #simple_perceptron()

    #descente_gradient()

    #adaline()
    
    #monocouche()

    #multicouche()

    sign_language()




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


def adaline():
    model_and = PerceptronAdaline(data_path=TABLE_AND, graph_name="Perceptron ADALINE Classification AND")

    model_and.train(learning_rate=0.1, epochs=500)

    model_and.display_graph_classification()

    model_2_9 = PerceptronAdaline(data_path=TABLE_2_9, graph_name="Perceptron ADALINE Classification linéairment séparable")

    model_2_9.train(0.001, 500)

    model_2_9.display_graph_classification()

    model_2_10 = PerceptronAdaline(data_path=TABLE_2_10, graph_name="Perceptron ADALINE Classification non linéairment séparable")

    model_2_10.train(0.001, 500)

    model_2_10.display_graph_classification()

    model_2_11 = PerceptronAdaline(data_path=TABLE_2_11, graph_name="Perceptron ADALINE Regression")

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

def multicouche():
    table_choice = input("Entrez le numéro de la table (1, 2, 3 ou 4) : ").strip()
    if table_choice == "1":
        model = MLPModulaire(
            data_path=TABLE_XOR,
            input_size=2,
            hidden_size=2,
            output_size=1,
            task="classification",
            output_activation="sigmoid"
        )

        model.train(learning_rate=0.1, epochs=15000)
        model.plot()

    if table_choice == "2":
        model = MLPModulaire(
            data_path=TABLE_4_12,
            input_size=2,
            hidden_size=4,  # Tu peux augmenter la complexité pour mieux apprendre
            output_size=1,
            task="classification",
            output_activation="sigmoid"
        )

        model.train(learning_rate=0.1, epochs=10000)
        model.plot()

    if table_choice == "3":
        model = MLPModulaire(
            data_path=TABLE_4_14,
            input_size=2,
            hidden_size=6,  # Plus de neurones cachés pour mieux séparer 3 classes
            output_size=3,
            task="classification",
            output_activation="softmax"
        )

        model.train(learning_rate=0.1, epochs=15000)
        model.plot()

    if table_choice == "4":
        model = MLPModulaire(
            data_path=TABLE_4_17,
            input_size=1,
            hidden_size=5,  # Suffisant pour capter non-linéarité
            output_size=1,
            task="regression",
            output_activation="linear"
        )

        model.train(learning_rate=0.01, epochs=20000)
        model.plot()

def sign_language():
    print("=== Classification Langage des Signes ===")


    data = LangageSignesData(data_path= TABLE_SIGN_LANGUAGE)
    data.summary()


    model = NeuralNetworkScratch(input_size=42, hidden_size=64, output_size=5, lr=0.01)


    print("Début de l'entraînement...")
    model.train(data.X_train, data.y_train, epochs=200)


    acc = model.accuracy(data.X_val, data.y_val)
    print(f"Accuracy sur validation : {acc * 100:.2f}%")


if __name__ == "__main__":
    main()

