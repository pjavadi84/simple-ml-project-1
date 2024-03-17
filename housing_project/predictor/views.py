from django.http import JsonResponse
from .model import predict_price  # Assume you've encapsulated your model logic in a function

def predict(request):
    # Example: fetch parameters from request for prediction
    # For a real application, you'd validate and extract these values properly
    sample_house = request.GET.get('sample_house', None)
    if sample_house:
        predicted_price = predict_price(sample_house)  # Your model function
        return JsonResponse({'predicted_price': predicted_price})
    else:
        return JsonResponse({'error': 'No input data provided'}, status=400)


