import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fonctions d’activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # stabilité numérique
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_derivative(x):
    return x * (1 - x)  # approximation pour usage simple

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

class MLPModulaire:
    def __init__(self, data_path, input_size, hidden_size, output_size, task='classification', output_activation='sigmoid'):
        self.task = task
        self.output_activation_name = output_activation

        # Chargement des données
        data = pd.read_csv(data_path, header=None).values
        self.X = data[:, :input_size]
        self.y = data[:, input_size:]

        # Normalisation pour classification binaire (-1 -> 0)
        if self.task == 'classification' and output_size == 1:
            self.y[self.y == -1] = 0

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialisation aléatoire
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

        # Activation de sortie
        self.output_activation = self.get_activation_function(output_activation)
        self.output_derivative = self.get_activation_derivative(output_activation)

    def get_activation_function(self, name):
        return {
            'sigmoid': sigmoid,
            'softmax': softmax,
            'linear': linear
        }[name]

    def get_activation_derivative(self, name):
        return {
            'sigmoid': sigmoid_derivative,
            'softmax': softmax_derivative,
            'linear': linear_derivative
        }[name]

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        if self.output_activation_name == 'linear':
            self.a1 = np.tanh(self.z1)
        else:
            self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.output_activation(self.z2)
        return self.a2

    def loss(self, y_true, y_pred):
        if self.task == 'regression':
            return np.mean((y_true - y_pred) ** 2)
        elif self.task == 'classification':
            return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
        return None

    def backward(self, X, y, output, learning_rate):
        output_error = y - output
        output_delta = output_error * self.output_derivative(output)

        hidden_error = np.dot(output_delta, self.W2.T)
        if self.output_activation_name == 'linear':
            hidden_delta = hidden_error * tanh_derivative(self.a1)  # dérivée tanh
        else:
            hidden_delta = hidden_error * sigmoid_derivative(self.a1)

        self.W2 += learning_rate * np.dot(self.a1.T, output_delta)
        self.b2 += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.W1 += learning_rate * np.dot(X.T, hidden_delta)
        self.b1 += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, learning_rate, epochs, verbose=True, plot_loss=True):
        losses = []

        for epoch in range(epochs):
            output = self.forward(self.X)
            loss = self.loss(self.y, output)
            losses.append(loss)

            self.backward(self.X, self.y, output, learning_rate)

            # Affichage console
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Époque {epoch + 1}/{epochs} - Erreur moyenne (loss) : {loss:.6f}")

        if plot_loss:
            plt.plot(range(1, epochs + 1), losses, color='purple')
            plt.title("Évolution de la fonction de perte")
            plt.xlabel("Époque")
            plt.ylabel("Erreur / perte")
            plt.grid(True)
            plt.show()

        print("Apprentissage terminé.")

        print("\nSorties finales après apprentissage :")
        outputs = self.forward(self.X)
        for i, (x, y_pred, y_true) in enumerate(zip(self.X, outputs, self.y)):
            print(f"Entrée {i + 1}: {x} → Prédiction: {y_pred[0]:.4f} (Attendu: {y_true[0]})")
        print(f"Poids finaux W1 :\n{self.W1}")
        print(f"Poids finaux W2 :\n{self.W2}")




    def predict(self, x):
        output = self.forward(np.array([x]))
        if self.task == 'classification':
            return np.argmax(output) if self.output_size > 1 else int(output[0][0] >= 0.5)
        else:
            return output[0][0]

    def plot(self):
        # Vérifie si c’est un problème à 2 dimensions
        if self.input_size != 2:
            if self.task == 'regression' and self.input_size == 1 and self.output_size == 1:
                # Visualisation pour la régression 1D
                x_vals = np.linspace(self.X.min(), self.X.max(), 300).reshape(-1, 1)
                y_preds = self.forward(x_vals)

                # Données réelles
                plt.scatter(self.X, self.y, color='blue', label='Données réelles', alpha=0.6)

                # Prédiction du modèle
                plt.plot(x_vals, y_preds, color='red', label='Prédiction du modèle', linewidth=2)

                plt.title("Régression non linéaire")
                plt.xlabel("Entrée")
                plt.ylabel("Sortie")
                plt.legend()
                plt.grid(True)
                plt.show()
                return
            else:
                print("Visualisation indisponible (entrée ≠ 2D et tâche ≠ régression 1D).")
                return

        h = 0.01
        x_min, x_max = self.X[:, 0].min() - 0.1, self.X[:, 0].max() + 0.1
        y_min, y_max = self.X[:, 1].min() - 0.1, self.X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid = np.c_[xx.ravel(), yy.ravel()]
        preds = self.forward(grid)

        # Multi-classes
        if self.output_size > 1:
            pred_classes = np.argmax(preds, axis=1)
            Z = pred_classes.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)

            colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
            true_classes = np.argmax(self.y, axis=1)
            unique_labels = np.unique(true_classes)

            for label in unique_labels:
                indices = np.where(true_classes == label)
                plt.scatter(self.X[indices][:, 0], self.X[indices][:, 1],
                            c=colors[label % len(colors)],
                            edgecolor='k',
                            label=f"Classe {label}")

            plt.title("Frontière de décision - Classification multi-classes")


        # Classification binaire
        elif self.output_size == 1:
            Z = preds.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
            plt.colorbar(label="Probabilité de sortie (classe 1)")

            unique_labels = np.unique(self.y)
            colors = ['blue', 'red']

            for i, label in enumerate(unique_labels):
                indices = np.where(self.y[:, 0] == label)
                plt.scatter(self.X[indices][:, 0], self.X[indices][:, 1],
                            c=colors[i % len(colors)],
                            edgecolor='k',
                            label=f"Classe {int(label)}")

            plt.title("Frontière de décision - Classification binaire")

        plt.xlabel("Entrée 1")
        plt.ylabel("Entrée 2")
        plt.legend()
        plt.grid(True)
        plt.show()

