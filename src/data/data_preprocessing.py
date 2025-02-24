import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        
        # Convertir les variables yes/no en 1/0
        binary_columns = ['mainroad', 'guestroom', 'basement', 
                         'hotwaterheating', 'airconditioning', 'prefarea']
        
        for column in binary_columns:
            df_processed[column] = (df_processed[column] == 'yes').astype(int)
            
        # Encoder furnishingstatus
        df_processed['furnishingstatus'] = self.label_encoder.fit_transform(
            df_processed['furnishingstatus']
        )
        
        # Normaliser les variables numériques
        numeric_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
        df_processed[numeric_columns] = self.scaler.fit_transform(
            df_processed[numeric_columns]
        )
        
        return df_processed
    
    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X = df.drop('price', axis=1)
        y = df['price']
        return X, y
    
    def get_feature_names(self) -> list:
        return ['area', 'bedrooms', 'bathrooms', 'stories', 'parking',
                'mainroad', 'guestroom', 'basement', 'hotwaterheating',
                'airconditioning', 'prefarea', 'furnishingstatus']