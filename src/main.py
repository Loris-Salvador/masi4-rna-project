from simple_perceptron.classification_simple import ClassificationSimplePerceptron
from simple_perceptron.linear_regression_simple import LinearRegressionSimplePerceptron
from perceptron_mono_couche.Perceptron3Classes import Perceptron3Classes
from perceptron_mono_couche.Perceptron4Classes import Perceptron4Classes

from data_path import TABLE_AND, TABLE_2_9, TABLE_2_10, TABLE_2_11, TABLE_3_1, TABLE_3_5



def main():
    print("Choisissez le type de modèle :")
    model_type = input("Entrez '1' pour classification, '2' pour régression ou '3' pour le monocouche : ").strip().lower()

    if model_type == '1':
        print("Choisissez la table pour la classification :")
        print("1. TABLE_AND (AND gate)")
        print("2. TABLE_2_9")
        print("3. TABLE_2_10")
        table_choice = input("Entrez le numéro de la table (1, 2, ou 3) : ").strip()
        if table_choice == "1":
            data_path = TABLE_AND
        elif table_choice == "2":
            data_path = TABLE_2_9
        elif table_choice == "3":
            data_path = TABLE_2_10
        else:
            print("Choix invalide, utilisation de la table TABLE_2_10 par défaut.")
            data_path = TABLE_2_10

        model = ClassificationSimplePerceptron(data_path=data_path)
    elif model_type == '2':
        print("Vous avez choisi la régression avec la table 2_11.")
        data_path = TABLE_2_11
        model = LinearRegressionSimplePerceptron(data_path=data_path)
    elif model_type == '3':
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
    else:
        print("Choix invalide. Quittez le programme.")
        return

    learning_rate = float(input("Entrez le taux d'apprentissage (par exemple 0.1) : "))
    epochs = int(input("Entrez le nombre d'époques (par exemple 500) : "))

    model.train(learning_rate=learning_rate, epochs=epochs)

    if model_type == '1':
        model.plot_decision_boundary()
    elif model_type == '3' and table_choice == '2':
        model.evaluate()
    else:
        model.plot()

if __name__ == "__main__":
    main()