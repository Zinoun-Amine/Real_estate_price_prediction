from zenml import pipeline, step
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

@step
def data_loading() -> pd.DataFrame:
    """Étape de chargement des données"""
    from src.data.data_loader import DataLoader
    loader = DataLoader()
    return loader.load_data()

@step
def data_preprocessing(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Étape de prétraitement des données"""
    from src.data.data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    data_processed = preprocessor.preprocess_data(data)
    X, y = preprocessor.split_features_target(data_processed)
    
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

@step
def model_training(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> dict:
    """Étape d'entraînement du modèle"""
    from src.models.model_training import ModelTrainer
    
    trainer = ModelTrainer()
    metrics = trainer.train_without_mlflow(X_train, y_train)
    
    # Validation croisée
    trainer.cross_validate(X_train, y_train)
    
    # Analyse des erreurs
    error_analysis = trainer.analyze_errors(X_test, y_test)
    
    return metrics

@pipeline
def real_estate_training_pipeline():
    """Pipeline complet d'entraînement"""
    # Chargement des données
    data = data_loading()
    
    # Prétraitement
    X_train, X_test, y_train, y_test = data_preprocessing(data)
    
    # Entraînement et évaluation
    metrics = model_training(X_train, X_test, y_train, y_test)