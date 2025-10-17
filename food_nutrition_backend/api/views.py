from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import onnxruntime as ort
import numpy as np
from PIL import Image
import json
import os
import requests
from dotenv import load_dotenv

# Load model and data on startup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load ONNX model
try:
    session = ort.InferenceSession(os.path.join(BASE_DIR, 'models', 'vgg16_food101.onnx'))
    print("✅ ONNX model loaded successfully")
except Exception as e:
    print(f"❌ Error loading ONNX model: {e}")
    session = None

# Load class names
with open(os.path.join(BASE_DIR, 'models', 'class_names.json'), 'r') as f:
    CLASS_NAMES = json.load(f)

# Load nutrition database
with open(os.path.join(BASE_DIR, 'models', 'nutrition_database.json'), 'r') as f:
    NUTRITION_DB = json.load(f)

# USDA API config
USDA_API_KEY = os.getenv('USDA_API_KEY')
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"


def preprocess_image(image):
    """Preprocess image for VGG16"""
    # Resize to 224x224
    image = image.resize((224, 224))
    
    # To numpy array
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Transpose to CHW format and add batch dimension
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    return img_array


def classify_image(image):
    """Classify food using ONNX model"""
    if session is None:
        raise Exception("Model not loaded")
    
    img_array = preprocess_image(image)
    
    # Run inference
    outputs = session.run(None, {'input': img_array})
    predictions = outputs[0][0]
    
    # Softmax
    exp_preds = np.exp(predictions - np.max(predictions))
    probabilities = exp_preds / exp_preds.sum()
    
    predicted_idx = np.argmax(probabilities)
    confidence = float(probabilities[predicted_idx] * 100)
    food_name = CLASS_NAMES[predicted_idx]
    
    return food_name, confidence


def get_nutrition(food_name):
    """Get nutrition with API + fallback"""
    # Try USDA API first
    try:
        search_url = f"{USDA_BASE_URL}/foods/search"
        params = {'api_key': USDA_API_KEY, 'query': food_name, 'pageSize': 1}
        response = requests.get(search_url, params=params, timeout=5)
        data = response.json()
        
        if data.get('foods'):
            food_item = data['foods'][0]
            nutrients = {'calories': 0, 'protein': 0, 'carbs': 0, 'fats': 0}
            
            for nutrient in food_item.get('foodNutrients', []):
                name = nutrient.get('nutrientName', '').lower()
                value = nutrient.get('value', 0)
                
                if 'energy' in name:
                    nutrients['calories'] = round(value, 1)
                elif 'protein' in name and 'amino' not in name:
                    nutrients['protein'] = round(value, 1)
                elif 'carbohydrate' in name:
                    nutrients['carbs'] = round(value, 1)
                elif 'total lipid' in name:
                    nutrients['fats'] = round(value, 1)
            
            if nutrients['calories'] > 0:
                return nutrients, 'usda_api'
    except Exception as e:
        print(f"API error: {e}")
    
    # Fallback to local database
    food_key = food_name.replace(' ', '_')
    nutrition = NUTRITION_DB.get(food_key, {'calories': 200, 'protein': 5, 'carbs': 25, 'fats': 8})
    return nutrition, 'local_database'


@api_view(['POST'])
def analyze_food(request):
    """
    Main endpoint for food analysis
    POST /api/analyze/
    Body: multipart/form-data with 'image' file
    """
    if 'image' not in request.FILES:
        return Response(
            {'error': 'No image provided'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        # Load image
        image_file = request.FILES['image']
        image = Image.open(image_file).convert('RGB')
        
        # Classify
        food_name, confidence = classify_image(image)
        
        # Get nutrition
        nutrition, source = get_nutrition(food_name)
        
        # Default portion (100g)
        portion_grams = 100
        
        # Build response
        result = {
            'success': True,
            'food_name': food_name,
            'confidence': round(confidence, 2),
            'portion_grams': portion_grams,
            'nutrition': {
                'calories': nutrition['calories'],
                'protein': nutrition['protein'],
                'carbs': nutrition['carbs'],
                'fats': nutrition['fats']
            },
            'data_source': source
        }
        
        return Response(result, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {'success': False, 'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def health_check(request):
    """Health check endpoint"""
    return Response({
        'status': 'healthy',
        'model_loaded': session is not None,
        'classes_loaded': len(CLASS_NAMES),
        'nutrition_db_loaded': len(NUTRITION_DB)
    })