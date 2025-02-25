import pandas as pd
import numpy as np
from src.models.model_training import ModelTrainer
from src.data.data_preprocessing import DataPreprocessor

class RealEstatePredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = DataPreprocessor()
        self._load_model()

    def _load_model(self):
        trainer = ModelTrainer()
        self.model = trainer.model

    def predict_price(self, 
                     area: float,
                     bedrooms: int,
                     bathrooms: int,
                     stories: int,
                     mainroad: str,
                     guestroom: str,
                     basement: str,
                     hotwaterheating: str,
                     airconditioning: str,
                     parking: int,
                     prefarea: str,
                     furnishing_status: str) -> dict:
        input_data = pd.DataFrame({
            'area': [float(area)],
            'bedrooms': [int(bedrooms)],
            'bathrooms': [int(bathrooms)],
            'stories': [int(stories)],
            'mainroad': [str(mainroad)],
            'guestroom': [str(guestroom)],
            'basement': [str(basement)],
            'hotwaterheating': [str(hotwaterheating)],
            'airconditioning': [str(airconditioning)],
            'parking': [int(parking)],
            'prefarea': [str(prefarea)],
            'furnishingstatus': [str(furnishing_status)]
        })

        try:
            # Prétraiter les données
            processed_data = self.preprocessor.preprocess_data(input_data)
            
            # Faire la prédiction
            prediction = self.model.predict(processed_data.drop('price', axis=1))
            
            # Convertir la prédiction
            predicted_price = np.exp(prediction[0])
            
            return {
                'predicted_price': float(predicted_price),
                'confidence_range': {
                    'lower_bound': float(predicted_price * 0.9),
                    'upper_bound': float(predicted_price * 1.1)
                }
            }
        except Exception as e:
            print(f"Erreur lors de la prédiction: {str(e)}")
            return {
                'error': str(e)
            }