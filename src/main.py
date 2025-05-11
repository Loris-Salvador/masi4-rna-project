from perceptron_mono_couche.Perceptron3Classes import Perceptron3Classes
from perceptron_mono_couche.Perceptron4Classes import Perceptron4Classes
from perceptron.perceptron_simple import   PerceptronSimple
from perceptron.perceptron_descente_gradient import PerceptronDescenteGradient
from perceptron.perceptron_adaline import PerceptronAdaline
from perceptron_multi_couche.MLPModulaire import MLPModulaire
from langage_des_signes.data_loader import LangageSignesData
from langage_des_signes.neural_network import NeuralNetworkScratch
import os

from data_path import TABLE_AND, TABLE_2_9, TABLE_2_10, TABLE_2_11, TABLE_3_1, TABLE_3_5, TABLE_XOR, TABLE_4_12, \
    TABLE_4_14, TABLE_4_17 , TABLE_SIGN_LANGUAGE


def main():
    os.system("cls")
    while True:

        print("0. Quitter")
        print("1. Perceptron Simple")
        print("2. Perceptron Descente Gradient")
        print("3. Perceptron Adaline")
        print("4. Perceptron Mono Couche")
        print("5. Perceptron Multi Couche")
        print("6. Perceptron Langage des signes")
        perceptron_choice = input("Choisissez le type de perceptron : ")

        if perceptron_choice == "0":
            break

        if perceptron_choice == "1" or perceptron_choice == "2" or perceptron_choice == "3":
            perceptron(perceptron_choice)

        elif perceptron_choice == "4":
            monocouche()

        elif perceptron_choice == "5":
            multicouche()

        elif perceptron_choice == "6":
            sign_language()

        os.system("cls")


def perceptron(perceptron_choice):
    data_path = perceptron_table_choice()
    graph_name = ""

    if perceptron_choice == "1":
        if data_path == TABLE_AND:
            graph_name = "Perceptron Simple Table AND"
        elif data_path == TABLE_2_9:
            graph_name = "Perceptron Simple données linéairement séparable"
        elif data_path == TABLE_2_10:
            graph_name = "Perceptron Simple données non linéairement séparable"
        elif data_path == TABLE_2_11:
            graph_name = "Perceptron Simple Regression linéaire"

        print("")
        ecart_type = input("Ecart type pour la génération des poids synaptiques : ")
        epochs = input("Entrez le nombre d'epochs : ")
        learning_rate = input("Entrez le learning rate : ")
        

        model = PerceptronSimple(data_path=data_path, graph_name=graph_name, ecart_type=float(ecart_type))

        model.train(epochs=int(epochs), learning_rate=float(learning_rate))

        if data_path == TABLE_2_11:
             model.display_graph_regression()
        else:
            model.display_graph_classification()


    elif perceptron_choice == "2":
        if data_path == TABLE_AND:
            graph_name = "Perceptron Descente Gradient Table AND"
        elif data_path == TABLE_2_9:
            graph_name = "Perceptron Descente Gradient données linéairement séparable"
        elif data_path == TABLE_2_10:
            graph_name = "Perceptron Descente Gradient données non linéairement séparable"
        elif data_path == TABLE_2_11:
            graph_name = "Perceptron Descente Gradient Regression linéaire"

        print("")
        ecart_type = input("Ecart type pour la génération des poids synaptiques : ")
        epochs = input("Entrez le nombre d'epochs : ")
        learning_rate = input("Entrez le learning rate : ")
        threshold = input("Entrez le threshold : ")

        model = PerceptronDescenteGradient(data_path=data_path, graph_name=graph_name, ecart_type=float(ecart_type))

        model.train(epochs=int(epochs), learning_rate=float(learning_rate), threshold=float(threshold))

        if data_path == TABLE_2_11:
            model.display_graph_regression()
        else:
            model.display_graph_classification()

    elif perceptron_choice == "3":
        if data_path == TABLE_AND:
            graph_name = "Perceptron ADALINE Table AND"
        elif data_path == TABLE_2_9:
            graph_name = "Perceptron ADALINE données linéairement séparable"
        elif data_path == TABLE_2_10:
            graph_name = "Perceptron ADALINE données non linéairement séparable"
        elif data_path == TABLE_2_11:
            graph_name = "Perceptron ADALINE Regression linéaire"

        print("")
        ecart_type = input("Ecart type pour la génération des poids synaptiques : ")
        epochs = input("Entrez le nombre d'epochs : ")
        learning_rate = input("Entrez le learning rate : ")
        threshold = input("Entrez le threshold : ")

        model = PerceptronAdaline(data_path=data_path, graph_name=graph_name, ecart_type=float(ecart_type))

        model.train(epochs=int(epochs), learning_rate=float(learning_rate), threshold=float(threshold))

        if data_path == TABLE_2_11:
            model.display_graph_regression()
        else:
            model.display_graph_classification()


def perceptron_table_choice():
    print("")
    print("1. Porte AND")
    print("2. Données linéairement séparable")
    print("3. Données non linéairement séparable")
    print("4. Regression linéaire")
    choice = input("Choisissez le type de donées : ")

    if choice == "1":
        return TABLE_AND
    elif choice == "2":
        return TABLE_2_9
    elif choice == "3":
        return TABLE_2_10
    elif choice == "4":
        return TABLE_2_11
    return None


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

        model.train(learning_rate=0.8, epochs=2000)
        model.plot()

    if table_choice == "2":
        model = MLPModulaire(
            data_path=TABLE_4_12,
            input_size=2,
            hidden_size=4,
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
            hidden_size=6,
            output_size=3,
            task="classification",
            output_activation="softmax",
        )

        model.train(learning_rate=0.1, epochs=15000)
        model.plot()

    if table_choice == "4":
        model = MLPModulaire(
            data_path=TABLE_4_17,
            input_size=1,
            hidden_size=5,
            output_size=1,
            task="regression",
            output_activation="linear"
        )

        model.train(learning_rate=0.004, epochs=3000)
        model.plot()

def sign_language():
    print("=== Classification Langage des Signes ===")


    data = LangageSignesData(data_path= TABLE_SIGN_LANGUAGE)
    data.summary()


    model = NeuralNetworkScratch(input_size=42, hidden_size=150, output_size=5, lr=0.15)


    print("Début de l'entraînement...")
    train_loss, val_loss = model.train(
        data.X_train, data.y_train,
        epochs=5000,
        X_val=data.X_val,
        y_val=data.y_val,
        tol=0.0001
    )

    y_pred = model.predict(data.X_train)
    model.plot_losses(train_loss, val_loss)
    model.plot_classification_2D(data.X_train, data.y_train, y_pred, title="Visualisation 2D des prédictions")




    acc = model.accuracy(data.X_val, data.y_val)
    print(f"Accuracy sur validation : {acc * 100:.2f}%")


if __name__ == "__main__":
    main()

