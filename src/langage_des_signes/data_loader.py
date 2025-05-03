import os
import pandas as pd
import numpy as np
import random


class LangageSignesData:
    def __init__(self, data_path):
        self.data_path = data_path
        self.class_names = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}
        self.load_data()
        self.prepare_datasets()

    def load_data(self):
        """Charge les données avec vérifications rigoureuses"""
        try:
            # Lecture avec vérification de format
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Fichier introuvable: {self.data_path}")

            data = pd.read_csv(self.data_path, header=None)

            # Vérification du shape
            if data.shape[0] != 300:
                raise ValueError(f"Nombre de lignes incorrect. Reçu {data.shape[0]}, attendu 300")

            if data.shape[1] not in [43, 47]:
                raise ValueError(f"Format non supporté. Reçu {data.shape[1]} colonnes (attendu 43 ou 47)")

            # Conversion des types
            self.X = data.iloc[:, :-1].astype(np.float32).values
            self.y = data.iloc[:, -1].astype(np.int32).values

            print(f"Données chargées: {len(self.X)} échantillons")

        except Exception as e:
            print(f"\nERREUR CRITIQUE dans load_data(): {str(e)}")
            print("Vérifiez que:")
            print("1. Le fichier existe bien à l'emplacement spécifié")
            print("2. Il contient exactement 300 lignes et 43 colonnes")
            print("3. Les valeurs sont bien numériques")
            raise

    def prepare_datasets(self, train_per_class=50, val_per_class=10):
        """Prépare les datasets avec stratification"""
        np.random.seed(42)

        X_train, y_train = [], []
        X_val, y_val = [], []

        for class_id in range(1, 6):
            class_indices = np.where(self.y == class_id)[0]

            if len(class_indices) < (train_per_class + val_per_class):
                raise ValueError(f"Classe {class_id} n'a pas assez d'échantillons")

            np.random.shuffle(class_indices)

            # Séparation train/val
            train_idx = class_indices[:train_per_class]
            val_idx = class_indices[train_per_class:(train_per_class + val_per_class)]

            X_train.append(self.X[train_idx])
            y_train.append(self.y[train_idx])
            X_val.append(self.X[val_idx])
            y_val.append(self.y[val_idx])

        # Concatenation et mélange final
        self.X_train = np.concatenate(X_train)
        self.y_train = np.concatenate(y_train)
        self.X_val = np.concatenate(X_val)
        self.y_val = np.concatenate(y_val)

        # Mélange
        train_shuffle = np.random.permutation(len(self.X_train))
        val_shuffle = np.random.permutation(len(self.X_val))

        self.X_train = self.X_train[train_shuffle]
        self.y_train = self.y_train[train_shuffle]
        self.X_val = self.X_val[val_shuffle]
        self.y_val = self.y_val[val_shuffle]

    def summary(self):
        """Affiche des statistiques détaillées"""
        print("\n=== DATA SUMMARY ===")
        print(f"Train set: {self.X_train.shape} (soit {len(self.X_train) / len(self.X) * 100:.1f}%)")
        print(f"Validation set: {self.X_val.shape} (soit {len(self.X_val) / len(self.X) * 100:.1f}%)")

        print("\nRépartition par classe:")
        for class_id in range(1, 6):
            train_count = np.sum(self.y_train == class_id)
            val_count = np.sum(self.y_val == class_id)
            print(f"{self.class_names[class_id]}: {train_count} train, {val_count} val")