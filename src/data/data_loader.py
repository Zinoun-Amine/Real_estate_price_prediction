# src/data/data_loader.py

import pandas as pd
import numpy as np
from typing import Tuple

class DataLoader:
    def __init__(self, data_path: str = 'data/raw/housing_data.csv'):
        self.data_path = data_path
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """
        Charge les données du fichier CSV
        """
        try:
            self.data = pd.read_csv(self.data_path)
            print("Données chargées avec succès!")
            print(f"Dimensions du dataset: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            return None

    def get_data_info(self):
        """
        Affiche les informations sur les données
        """
        if self.data is not None:
            print("\n=== Information sur les données ===")
            print("\nAperçu des données:")
            print(self.data.head())
            print("\nInformations sur les colonnes:")
            print(self.data.info())
            print("\nStatistiques descriptives:")
            print(self.data.describe())
            print("\nValeurs manquantes:")
            print(self.data.isnull().sum())