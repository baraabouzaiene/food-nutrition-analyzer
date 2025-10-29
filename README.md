# 🍕 Food Nutrition Analyzer

**AI-powered food recognition system using computer vision and deep learning to identify food items and provide detailed nutritional information.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.2-green.svg)](https://www.djangoproject.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange.svg)](https://onnxruntime.ai/)

[**📹 Watch Demo Video**](#) | [**🔗 GitHub Repository**](https://github.com/baraabouzaiene/food-nutrition-analyzer)

---

## 🎯 Overview

Upload a food image and instantly get AI-powered classification with confidence scores and complete nutritional breakdown. Built with VGG16 convolutional neural network achieving 75% accuracy on 101 food categories, optimized for production with ONNX Runtime.

### Key Features

- 🎯 **Computer Vision Classification** - VGG16 CNN trained on Food-101 dataset
- 📊 **Top-3 Predictions** - Alternative classifications with confidence scores
- 🥗 **Real-time Nutrition Data** - Integration with USDA FoodData Central API
- ⚖️ **Adjustable Portions** - Dynamic nutrition calculation for custom serving sizes
- 🔄 **Fallback System** - Local database ensures 100% uptime
- 🚀 **Optimized Inference** - ONNX Runtime for 60% faster predictions

---

## 🏗️ Architecture

```
Frontend (HTML/JS)
       ↓
Django REST API
       ↓
   ┌───┴───┐
   ↓       ↓
VGG16    USDA API
ONNX     + Local DB
```

### Tech Stack

**Machine Learning & Computer Vision:**
- PyTorch - Model training and development
- VGG16 - Convolutional neural network architecture
- ONNX Runtime - Optimized model inference
- PIL & NumPy - Image preprocessing pipeline

**Backend:**
- Django 5.2 + Django REST Framework
- Python 3.8+
- Requests - API integration

**Frontend:**
- Vanilla JavaScript (Fetch API)
- HTML5 + CSS3

**External APIs:**
- USDA FoodData Central

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip
Virtual environment (recommended)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/baraabouzaiene/food-nutrition-analyzer.git
cd food-nutrition-analyzer
```

2. **Set up backend**
```bash
cd food_nutrition_backend
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
# Create .env file in food_nutrition_backend/
echo "USDA_API_KEY=your_api_key_here" > .env
```

Get your free USDA API key: https://fdc.nal.usda.gov/api-key-signup.html

4. **Run the server**
```bash
python manage.py runserver
```

5. **Open the application**
```bash
# Open index.html in your browser
# Or navigate to http://localhost:8000
```

---

## 📖 API Documentation

### POST `/api/analyze/`

Analyze food image and return predictions with nutrition information.

**Request:**
```bash
curl -X POST http://localhost:8000/api/analyze/ \
  -F "image=@pizza.jpg" \
  -F "portion_grams=150"
```

**Parameters:**
- `image` (file, required) - Food image (JPG, PNG)
- `portion_grams` (integer, optional) - Portion size in grams (default: 100)

**Response:**
```json
{
  "success": true,
  "predictions": [
    {"food_name": "pizza", "confidence": 85.23},
    {"food_name": "flatbread", "confidence": 8.12},
    {"food_name": "calzone", "confidence": 3.45}
  ],
  "top_prediction": "pizza",
  "confidence": 85.23,
  "portion_grams": 150,
  "nutrition": {
    "calories": 399.0,
    "protein": 16.5,
    "carbs": 49.5,
    "fats": 15.0
  },
  "data_source": "usda_api"
}
```

**Status Codes:**
- `200` - Success
- `400` - Bad request (missing image)
- `500` - Server error

### GET `/api/health/`

Check API health and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "classes_loaded": 101,
  "nutrition_db_loaded": 101
}
```

---

## 📓 Model Training

Training notebooks are available in the [`notebooks/`](notebooks/) directory:

- `01_data_exploration.ipynb` - Food-101 dataset analysis and visualization
- `02_model_training.ipynb` - VGG16 training pipeline and hyperparameters
- `03_model_evaluation.ipynb` - Performance metrics and error analysis
- `04_pytorch_to_onnx.ipynb` - Model conversion and optimization

### Dataset: Food-101

- **Classes:** 101 food categories
- **Images:** 101,000 total (1,000 per class)
- **Split:** 75,750 training / 25,250 test
- **Source:** Real-world images with varying conditions

### Model Performance

- **Top-1 Accuracy:** ~75%
- **Top-3 Accuracy:** ~92%
- **Inference Time:** 50-100ms per image (ONNX)
- **Model Size:** ~500MB (ONNX vs 1GB PyTorch)

---

## 🔬 Technical Decisions

### Why VGG16?

- ✅ Proven accuracy on Food-101 benchmark
- ✅ Strong transfer learning from ImageNet
- ✅ Balance of performance vs computational cost
- ✅ Well-documented architecture

### Why ONNX Runtime?

- ⚡ **60% faster** inference than PyTorch
- 💾 **50% smaller** model size
- 🌐 Platform-agnostic deployment
- 🔧 Production-ready optimization

### Portion Size Approach

**Evaluated three approaches:**

1. ✅ **User Input** (Current) - Simple, reliable, accurate
2. ❌ **Image Segmentation** - Tested DeepLabV3, achieved only 26% coverage
3. 🔮 **Reference Object Detection** - Would require YOLO training (future work)

**Decision:** Prioritized reliability and user experience. Computer vision-based portion estimation proved unreliable in testing, with pixel area failing to correlate with actual weight due to density variations.

### Nutrition Data Strategy

- **Primary:** USDA FoodData Central API (accurate, up-to-date)
- **Fallback:** Local JSON database (ensures availability)
- **Timeout:** 5 seconds to prevent hanging requests
- **Transparency:** Response indicates data source

---

## 🐛 Known Limitations

- **Dataset Bias** - Food-101 underrepresents non-Western cuisines
- **Single Item Only** - Cannot detect multiple foods in one image
- **Lighting Sensitivity** - Performance degrades in poor lighting
- **Similar Foods** - May confuse visually similar items (e.g., pizza variations)

---

## 🚀 Future Enhancements

**Near-term:**
- [ ] Add more global cuisines to training data
- [ ] Implement image quality validation
- [ ] Add micronutrient information (vitamins, minerals)

**Medium-term:**
- [ ] User accounts with meal history tracking
- [ ] Mobile app (iOS/Android)
- [ ] Barcode scanning for packaged foods

**Long-term:**
- [ ] Multi-food detection (YOLO integration)
- [ ] Reference object detection for automatic portion sizing
- [ ] Integration with fitness tracking platforms

---

## 📁 Project Structure

```
food-nutrition-analyzer/
├── food_nutrition_backend/
│   ├── api/
│   │   ├── views.py              # API endpoints and logic
│   │   └── urls.py               # URL routing
│   ├── backend/
│   │   ├── settings.py           # Django configuration
│   │   └── urls.py               # Main URL router
│   ├── models/
│   │   ├── vgg16_food101.onnx           # Trained ONNX model
│   │   ├── class_names.json             # 101 food categories
│   │   └── nutrition_database.json      # Fallback nutrition data
│   ├── manage.py
│   └── requirements.txt
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_model_evaluation.ipynb
│   └── 04_pytorch_to_onnx.ipynb
├── frontend.html                    
└── README.md
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**[Your Name]**
- GitHub: (https://github.com/baraabouzaiene)
- LinkedIn: www.linkedin.com/in/baraa-bouzaiene-75727b246


- Email: baraabouzayen@gmail.com

---

## 🙏 Acknowledgments

- Food-101 dataset by Bossard et al.
- USDA FoodData Central API
- PyTorch and ONNX communities
- VGG16 architecture by Visual Geometry Group, Oxford

---

**Built with ❤️ and deep learning**
