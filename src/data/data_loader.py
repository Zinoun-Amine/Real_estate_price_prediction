import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, data_path: str = 'data/raw/housing_data.csv'):
        self.data_path = data_path
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """
        Charge les données depuis le fichier CSV
        """
        try:
            self.data = pd.read_csv(self.data_path)
            print("Données chargées avec succès!")
            print(f"Dimensions du dataset: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            return None