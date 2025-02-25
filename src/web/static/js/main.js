document.addEventListener('DOMContentLoaded', function() {
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePredictionSubmit);
    }
});

async function handlePredictionSubmit(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        // Afficher les résultats
        document.getElementById('predictionResult').style.display = 'block';
        document.getElementById('predictedPrice').textContent = 
            new Intl.NumberFormat('fr-FR', { 
                style: 'currency', 
                currency: 'INR' 
            }).format(result.predicted_price);
        
        document.getElementById('confidenceInterval').textContent = 
            `Entre ${new Intl.NumberFormat('fr-FR', { 
                style: 'currency', 
                currency: 'INR' 
            }).format(result.confidence_range.lower_bound)} et ${
                new Intl.NumberFormat('fr-FR', { 
                    style: 'currency', 
                    currency: 'INR' 
                }).format(result.confidence_range.upper_bound)}`;

    } catch (error) {
        console.error('Erreur:', error);
        alert('Une erreur est survenue lors de la prédiction');
    }
}