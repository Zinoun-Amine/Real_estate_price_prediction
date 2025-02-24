# main.py

from src.data.data_loader import DataLoader
from src.data.data_preprocessing import DataPreprocessor
from src.models.model_training import ModelTrainer
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import subprocess

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_experiment_tracking():
    """Configure le suivi des expériences"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/experiment_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def save_results(metrics: dict, feature_importance: pd.DataFrame, results_dir: str):
    """Sauvegarde les résultats de l'expérience"""
    # Sauvegarder les métriques
    pd.DataFrame([metrics]).to_csv(f"{results_dir}/metrics.csv", index=False)
    
    # Sauvegarder l'importance des features
    feature_importance.to_csv(f"{results_dir}/feature_importance.csv", index=False)

def initialize_zenml():
    """Initialise ZenML"""
    try:
        # Vérifier si ZenML est déjà initialisé
        result = subprocess.run(['zenml', 'status'], capture_output=True, text=True)
        if "ZenML repository is not initialized" in result.stderr:
            subprocess.run(['zenml', 'init'], check=True)
            logger.info("ZenML initialisé avec succès")
        else:
            logger.info("ZenML déjà initialisé")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de ZenML: {str(e)}")
        return False

def main():
    try:
        # Initialiser ZenML
        if not initialize_zenml():
            logger.warning("Continuer sans ZenML...")

        # Créer le dossier pour les résultats
        results_dir = setup_experiment_tracking()
        logger.info("Démarrage de l'expérience")

        # 1. Chargement des données
        logger.info("=== Chargement des données ===")
        loader = DataLoader()
        data = loader.load_data()
        
        if data is not None:
            # 2. Prétraitement
            logger.info("\n=== Prétraitement des données ===")
            preprocessor = DataPreprocessor()
            data_processed = preprocessor.preprocess_data(data)
            
            # 3. Split features/target
            X, y = preprocessor.split_features_target(data_processed)
            logger.info(f"\nDimensions des données:")
            logger.info(f"Features (X): {X.shape}")
            logger.info(f"Target (y): {y.shape}")
            
            # 4. Entraînement du modèle
            logger.info("\n=== Entraînement du modèle ===")
            trainer = ModelTrainer()
            metrics = trainer.train_without_mlflow(X, y)
            
            # 5. Validation croisée
            trainer.cross_validate(X, y)
            
            # 6. Analyse des erreurs
            error_analysis = trainer.analyze_errors(X, y)
            logger.info("\n=== Top 10 des plus grandes erreurs ===")
            logger.info(error_analysis)
            
            # 7. Exemple de prédiction
            logger.info("\n=== Exemple de prédiction ===")
            sample = X.iloc[0:1]
            prediction = trainer.predict(sample)
            real_value = y.iloc[0]
            logger.info(f"Valeur réelle: {np.exp(real_value):,.2f}")
            logger.info(f"Valeur prédite: {np.exp(prediction[0]):,.2f}")

            # 8. Sauvegarder les résultats
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': trainer.model.feature_importances_
            }).sort_values('importance', ascending=False)

            save_results(metrics, feature_importance, results_dir)
            logger.info(f"\nRésultats sauvegardés dans: {results_dir}")

            # 9. Résumé des performances
            logger.info("\n=== Résumé des performances ===")
            logger.info(f"R² Score (Test): {metrics['test_r2']:.4f}")
            logger.info(f"RMSE (Test): {metrics['test_rmse']:.4f}")
            logger.info(f"MAE (Test): {metrics['test_mae']:.4f}")

        else:
            logger.error("Erreur lors du chargement des données")

    except Exception as e:
        logger.error(f"Une erreur s'est produite: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()