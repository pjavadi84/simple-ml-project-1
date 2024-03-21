from django.http import JsonResponse
import numpy as np
import joblib
import pdb
import os
from .models import train_model

def predict(request):
    
    try:
        # Load the model from the file
        model_path = os.path.join(os.path.dirname(__file__), '..', 'linear_regression_model.joblib')
        model = joblib.load(model_path)
        # Fetch parameters from request for prediction (this is just an example)
        # You need to parse the input into the expected format
        sample_house = request.GET.get('sample_house', None)

        if sample_house is None:
            return JsonResponse({'error': 'No sample_house parameter provided'}, status=400)
        
        try:
            sample_house = np.fromstring(sample_house, sep=',').reshape(1, -1)
        except ValueError as e:
            return JsonResponse({'error': 'Invalid sample_house parameter format'}, status=400)

        
        if sample_house.shape == (1, 13):  # Assuming we need 13 features
            predicted_price = model.predict(sample_house)
            return JsonResponse({'predicted_price': predicted_price[0]})
        else:
            return JsonResponse({'error': 'Invalid input shape'}, status=400)
            
    except FileNotFoundError:
        return JsonResponse({'error': 'Model file not found'}, status=500)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
