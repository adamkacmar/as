from flask import Flask, request, jsonify
import time
from PIL import Image
import io

app = Flask(__name__)

CLASS_NAMES = ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]
DEVICE = "cpu"

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'device': DEVICE
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Mock prediction endpoint with metrics"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    try:
        total_start = time.time()
        preprocess_start = time.time()
        
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        image.thumbnail((256, 256))
        
        preprocess_time = (time.time() - preprocess_start) * 1000
        
        inference_start = time.time()
        time.sleep(0.05)
        import random
        predicted_idx = random.randint(0, 3)
        confidence = random.uniform(0.7, 0.99)
        inference_time = (time.time() - inference_start) * 1000
        
        total_time = (time.time() - total_start) * 1000
        
        return jsonify({
            'success': True,
            'predicted_class': CLASS_NAMES[predicted_idx],
            'confidence': float(confidence),
            'top_predictions': [
                {'class': CLASS_NAMES[i], 'confidence': float(max(0, confidence - i*0.15))}
                for i in range(min(3, len(CLASS_NAMES)))
            ],
            'metrics': {
                'preprocessing_time_ms': round(preprocess_time, 2),
                'inference_time_ms': round(inference_time, 2),
                'total_time_ms': round(total_time, 2),
                'device': DEVICE
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({
        'classes': CLASS_NAMES,
        'num_classes': len(CLASS_NAMES)
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Swin V2 API - Demo Version (No Model)")
    print("="*60)
    print(f"API running on http://localhost:5000")
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    print(f"\nEndpoints:")
    print(f"  - GET  /health    - Health check")
    print(f"  - POST /predict   - Mock image classification with metrics")
    print(f"  - GET  /classes   - List available classes")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
