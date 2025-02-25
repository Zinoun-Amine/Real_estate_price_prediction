from flask import Flask, render_template, jsonify, request
from src.prediction.predictor import RealEstatePredictor
import pandas as pd
import os

app = Flask(__name__, 
    template_folder='src/web/templates',
    static_folder='src/web/static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        data = request.json
        print("Données reçues:", data)  
        
        predictor = RealEstatePredictor()
        result = predictor.predict_price(**data)
        
        print("Résultat:", result) 
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
            
        return jsonify(result)
    except Exception as e:
        print("Erreur:", str(e))  
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis-data')
def analysis_data():
    # Charger les données pour les graphiques
    data = pd.read_csv('data/raw/housing_data.csv')
    feature_importance = pd.read_csv('results/feature_importance.csv')

    return jsonify({
        'price_distribution': {
            'labels': list(data['price'].value_counts().index),
            'values': list(data['price'].value_counts().values)
        },
        'feature_importance': {
            'features': list(feature_importance['feature']),
            'importance': list(feature_importance['importance'])
        }
    })

if __name__ == '__main__':
    app.run(debug=True)