import numpy as np
import pandas as pd
import os



class SimplePerceptronBase():

    def __init__(self, data_path):
        data = pd.read_csv(data_path, header=None)

        #toutes les lignes et toutes les colonnes sauf derniere
        self.entries = data.iloc[:, :-1].values

        #toutes les lignes et uniquement la derniere colonne
        self.exp_outputs = data.iloc[:, -1].values 

        # Ajouter une colonne de 1 pour le biais à self.entries
        # shape[0] = ligne et shape[1] = colonne
        self.entries = np.c_[np.ones((self.entries.shape[0], 1)), self.entries]

        self.weights = np.random.randn(self.entries.shape[1])