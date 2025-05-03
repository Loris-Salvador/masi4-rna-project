import pandas as pd
import numpy as np
import random


class LangageSignesData:
    def __init__(self, data_path):
        self.data_path = data_path
        self.load_data()
        self.prepare_datasets()

    def load_data(self):
        """Charge les données depuis le fichier CSV"""
        data = pd.read_csv(self.data_path)
        self.X = data.iloc[:, 1:].values  # Features (42 colonnes)
        self.y = data.iloc[:, 0].values  # Classes (1-5)
        self.class_names = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}

    def prepare_datasets(self, train_per_class=50, val_per_class=10, random_seed=42):
        """Sépare les données en ensembles d'apprentissage et de validation"""
        np.random.seed(random_seed)
        random.seed(random_seed)

        X_train, y_train = [], []
        X_val, y_val = [], []

        # Séparation stratifiée par classe
        for class_label in range(1, 6):
            indices = np.where(self.y == class_label)[0]
            np.random.shuffle(indices)

            # 50 pour train, 10 pour val par classe
            X_train.extend(self.X[indices[:train_per_class]])
            y_train.extend(self.y[indices[:train_per_class]])
            X_val.extend(self.X[indices[train_per_class:train_per_class + val_per_class]])
            y_val.extend(self.y[indices[train_per_class:train_per_class + val_per_class]])

        # Conversion en arrays numpy
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_val = np.array(X_val)
        self.y_val = np.array(y_val)

        # Mélange final
        train_shuffle = np.random.permutation(len(self.X_train))
        val_shuffle = np.random.permutation(len(self.X_val))

        self.X_train = self.X_train[train_shuffle]
        self.y_train = self.y_train[train_shuffle]
        self.X_val = self.X_val[val_shuffle]
        self.y_val = self.y_val[val_shuffle]

    def summary(self):
        """Affiche un résumé des données"""
        print(f"Train set: {self.X_train.shape}, Validation set: {self.X_val.shape}")
        print(f"Classes: {np.unique(self.y_train)}")
