from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.metrics = {}

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        # Split des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Entraînement avec MLflow tracking
        with mlflow.start_run():
            # Log des paramètres
            mlflow.log_params(self.model.get_params())

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

            # Log des métriques
            mlflow.log_metrics(self.metrics)

            # Log du modèle
            mlflow.sklearn.log_model(self.model, "random_forest_model")

            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\n=== Résultats de l'entraînement ===")
            print(f"R² Score (Train): {self.metrics['train_r2']:.4f}")
            print(f"R² Score (Test): {self.metrics['test_r2']:.4f}")
            print(f"RMSE (Train): {self.metrics['train_rmse']:.2f}")
            print(f"RMSE (Test): {self.metrics['test_rmse']:.2f}")
            print(f"MAE (Train): {self.metrics['train_mae']:.2f}")
            print(f"MAE (Test): {self.metrics['test_mae']:.2f}")
            
            print("\n=== Feature Importance ===")
            print(feature_importance)

        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5):
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        print(f"\n=== Validation Croisée (CV={cv}) ===")
        print(f"Scores R² : {cv_scores}")
        print(f"Moyenne : {cv_scores.mean():.4f}")
        print(f"Écart-type : {cv_scores.std():.4f}")