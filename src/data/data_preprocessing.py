import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prétraite les données
        """
        df_processed = df.copy()
        
        # 1. Log transformation du prix et de la surface
        df_processed['price'] = np.log1p(df_processed['price'])
        df_processed['area'] = np.log1p(df_processed['area'])
        
        # 2. Convertir les variables yes/no en 1/0
        binary_columns = ['mainroad', 'guestroom', 'basement', 
                         'hotwaterheating', 'airconditioning', 'prefarea']
        
        for column in binary_columns:
            df_processed[column] = (df_processed[column] == 'yes').astype(int)
            
        # 3. Encoder furnishingstatus
        df_processed['furnishingstatus'] = self.label_encoder.fit_transform(
            df_processed['furnishingstatus']
        )
        
        # 4. Créer des features plus sophistiquées
        df_processed['rooms_per_floor'] = df_processed['bedrooms'] / np.maximum(df_processed['stories'], 1)
        df_processed['total_rooms'] = df_processed['bedrooms'] + df_processed['bathrooms']
        df_processed['area_per_room'] = df_processed['area'] / df_processed['total_rooms']
        df_processed['bathrooms_ratio'] = df_processed['bathrooms'] / df_processed['bedrooms']
        df_processed['total_amenities'] = (df_processed[binary_columns].sum(axis=1) + 
                                         df_processed['parking'])
        
        # 5. Interactions
        df_processed['area_stories'] = df_processed['area'] * df_processed['stories']
        df_processed['area_bathrooms'] = df_processed['area'] * df_processed['bathrooms']
        
        # 6. Normaliser les variables numériques
        numeric_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking',
                         'rooms_per_floor', 'total_rooms', 'area_per_room',
                         'bathrooms_ratio', 'total_amenities', 'area_stories',
                         'area_bathrooms']
        
        df_processed[numeric_columns] = self.scaler.fit_transform(
            df_processed[numeric_columns]
        )
        
        return df_processed

    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Sépare les features et la variable cible
        """
        X = df.drop('price', axis=1)
        y = df['price']
        return X, y