# main.py

from src.data.data_loader import DataLoader
from src.data.data_preprocessing import DataPreprocessor
from src.models.model_training import ModelTrainer
import pandas as pd
import mlflow

def main():
    # Configuration de MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("real_estate_prediction")

    # Charger les données
    loader = DataLoader()
    data = loader.load_data()
    
    if data is not None:
        # Prétraiter les données
        preprocessor = DataPreprocessor()
        data_processed = preprocessor.preprocess_data(data)
        
        # Séparer features et target
        X, y = preprocessor.split_features_target(data_processed)
        
        # Entraîner le modèle
        trainer = ModelTrainer()
        
        # Entraînement et évaluation
        metrics = trainer.train(X, y)
        
        # Validation croisée
        trainer.cross_validate(X, y)
        
        # Exemple de prédiction
        print("\n=== Exemple de prédiction ===")
        sample_prediction = trainer.predict(X.head(1))
        print(f"Prix réel: {y.iloc[0]:,.2f}")
        print(f"Prix prédit: {sample_prediction[0]:,.2f}")

if __name__ == "__main__":
    main()