import numpy as np
import pandas as pd


class SimplePerceptronBase:
    def __init__(self, data_path):
        data = pd.read_csv(data_path, header=None)

        #toutes les lignes et toutes les colonnes sauf derniere
        self.entries = data.iloc[:, :-1].values

        #toutes les lignes et uniquement la derniere colonne
        self.exp_outputs = data.iloc[:, -1].values 

        # Ajouter une colonne de 1 pour le biais à self.entries
        self.entries = np.c_[np.ones((self.entries.shape[0], 1)), self.entries]

        #shape[0] donne le nb lignes et shape[1] le nb colonnes
        n_features = self.entries.shape[1]

        self.weights = np.random.randn(n_features)
