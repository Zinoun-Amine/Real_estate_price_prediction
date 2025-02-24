from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        self.metrics = {}

    def train_without_mlflow(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """
        Entraîne le modèle sans MLflow
        """
        # Split des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Entraînement du modèle
        self.model.fit(X_train, y_train)

        # Prédictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        # Calcul des métriques
        self.metrics = {
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test)
        }

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n=== Résultats de l'entraînement ===")
        print(f"R² Score (Train): {self.metrics['train_r2']:.4f}")
        print(f"R² Score (Test): {self.metrics['test_r2']:.4f}")
        print(f"RMSE (Train): {self.metrics['train_rmse']:.4f}")
        print(f"RMSE (Test): {self.metrics['test_rmse']:.4f}")
        print(f"MAE (Train): {self.metrics['train_mae']:.4f}")
        print(f"MAE (Test): {self.metrics['test_mae']:.4f}")
        
        print("\n=== Feature Importance ===")
        print(feature_importance)

        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fait des prédictions sur de nouvelles données
        """
        return self.model.predict(X)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5):
        """
        Effectue une validation croisée
        """
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        print(f"\n=== Validation Croisée (CV={cv}) ===")
        print(f"Scores R² : {cv_scores}")
        print(f"Moyenne : {cv_scores.mean():.4f}")
        print(f"Écart-type : {cv_scores.std():.4f}")

    def analyze_errors(self, X: pd.DataFrame, y: pd.Series):
        """
        Analyse détaillée des erreurs de prédiction
        """
        predictions = self.model.predict(X)
        errors = y - predictions
        
        print("\n=== Analyse des erreurs ===")
        print(f"Erreur moyenne: {errors.mean():.2f}")
        print(f"Erreur médiane: {np.median(errors):.2f}")
        print(f"Écart-type des erreurs: {errors.std():.2f}")
        
        # Identifier les cas problématiques
        large_errors = np.abs(errors) > (2 * errors.std())
        print(f"\nNombre de prédictions avec grandes erreurs: {sum(large_errors)}")
        
        return pd.DataFrame({
            'real': y,
            'predicted': predictions,
            'error': errors,
            'abs_error': np.abs(errors)
        }).sort_values('abs_error', ascending=False).head(10)