from django.http import JsonResponse
import numpy as np
import joblib
import pdb

def predict(request):
    try:
        # Load the model from the file
        model = joblib.load('linear_regression_model.joblib')
        # Fetch parameters from request for prediction (this is just an example)
        # You need to parse the input into the expected format
        sample_house = request.GET.get('sample_house', None)
        
        # Convert string of numbers into a numpy array
        sample_house = np.fromstring(sample_house, sep=',').reshape(1, -1)
        
        if sample_house.shape == (1, 13):  # Assuming we need 13 features
            predicted_price = model.predict(sample_house)
            return JsonResponse({'predicted_price': predicted_price[0]})
        else:
            return JsonResponse({'error': 'Invalid input shape'}, status=400)
            
    except FileNotFoundError:
        return JsonResponse({'error': 'Model file not found'}, status=500)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
