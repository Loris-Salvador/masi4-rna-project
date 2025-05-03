import numpy as np

class NeuralNetworkScratch:
    def __init__(self, input_size=42, hidden_size=64, output_size=5, lr=0.01):
        self.lr = lr

        # Initialisation des poids et biais
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # stabilité numérique
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true])
        return np.sum(log_likelihood) / m

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y_true):
        m = X.shape[0]
        y_pred = self.a2
        y_one_hot = np.zeros_like(y_pred)
        y_one_hot[np.arange(m), y_true - 1] = 1  # classes de 1 à 5

        dz2 = (y_pred - y_one_hot) / m
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=100, X_val=None, y_val=None):
        train_losses = []  # Stocke les losses d'entraînement
        val_losses = []  # Stocke les losses de validation

        for epoch in range(epochs):
            # Entraînement normal
            y_pred = self.forward(X)
            loss = self.cross_entropy_loss(y_pred, y - 1)
            self.backward(X, y)
            train_losses.append(loss)

            # Calcul de la validation loss si données fournies
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.cross_entropy_loss(val_pred, y_val - 1)
                val_losses.append(val_loss)

            # Affichage périodique
            if (epoch + 1) % 10 == 0:
                val_log = f", Val Loss: {val_losses[-1]:.4f}" if val_losses else ""
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss:.4f}{val_log}")

        return train_losses, val_losses  # Retourne les courbes


    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1) + 1  # on retourne les classes de 1 à 5

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

    def plot_losses(self, train_losses, val_losses=None):
        import matplotlib.pyplot as plt
        plt.plot(train_losses, label="Train Loss")
        if val_losses:
            plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()