from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
import time
from transformers import Swinv2ForImageClassification

app = Flask(__name__)

MODEL_PATH = "model_lr1e-05_bs32.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

CLASS_NAMES = ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]

model = None

def load_model():
    global model
    
    print(f"Loading Swin V2 model from {MODEL_PATH}...")
    
    model = Swinv2ForImageClassification.from_pretrained(
        "microsoft/swinv2-base-patch4-window8-256",
        num_labels=len(CLASS_NAMES),
        ignore_mismatched_sizes=True
    )
    
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()
    
    print(f"Swin V2 model loaded successfully on {DEVICE}")
    print(f"Number of classes: {len(CLASS_NAMES)}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    
    return model


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(DEVICE)
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        total_start = time.time()
        
        preprocess_start = time.time()
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        preprocess_time = (time.time() - preprocess_start) * 1000  
        
        inference_start = time.time()
        with torch.no_grad():
            outputs = model(img_tensor)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        inference_time = (time.time() - inference_start) * 1000 
        
        predicted_class = CLASS_NAMES[predicted_idx.item()] if predicted_idx.item() < len(CLASS_NAMES) else f"class_{predicted_idx.item()}"
        confidence_score = confidence.item()
        
        top_k = min(3, len(CLASS_NAMES))
        top_probs, top_indices = torch.topk(probabilities[0], top_k)
        
        top_predictions = [
            {
                'class': CLASS_NAMES[idx.item()] if idx.item() < len(CLASS_NAMES) else f"class_{idx.item()}",
                'confidence': float(prob.item())
            }
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        total_time = (time.time() - total_start) * 1000  
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': float(confidence_score),
            'top_predictions': top_predictions,
            'metrics': {
                'preprocessing_time_ms': round(preprocess_time, 2),
                'inference_time_ms': round(inference_time, 2),
                'total_time_ms': round(total_time, 2),
                'device': str(DEVICE)
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/classes', methods=['GET'])
def get_classes():
    """Return available class names"""
    return jsonify({
        'classes': CLASS_NAMES,
        'num_classes': len(CLASS_NAMES)
    })


@app.route('/benchmark', methods=['POST'])
def benchmark():
    """
    Benchmark endpoint for measuring average inference time
    Expects: multipart/form-data with multiple 'images' files
    Returns: Aggregated statistics of inference times
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    files = request.files.getlist('images')
    
    if not files:
        return jsonify({'error': 'No images provided'}), 400
    
    results = {
        'total_images': len(files),
        'successful': 0,
        'failed': 0,
        'metrics': {
            'preprocessing_times_ms': [],
            'inference_times_ms': [],
            'total_times_ms': []
        },
        'device': str(DEVICE)
    }
    
    try:
        for file in files:
            if file.filename == '':
                results['failed'] += 1
                continue
            
            try:
                total_start = time.time()
                
                preprocess_start = time.time()
                img_bytes = file.read()
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                img_tensor = transform(image).unsqueeze(0).to(DEVICE)
                preprocess_time = (time.time() - preprocess_start) * 1000
                
                inference_start = time.time()
                with torch.no_grad():
                    outputs = model(img_tensor)
                inference_time = (time.time() - inference_start) * 1000
                
                total_time = (time.time() - total_start) * 1000
                
                results['metrics']['preprocessing_times_ms'].append(round(preprocess_time, 2))
                results['metrics']['inference_times_ms'].append(round(inference_time, 2))
                results['metrics']['total_times_ms'].append(round(total_time, 2))
                results['successful'] += 1
                
            except Exception as e:
                results['failed'] += 1
        
        if results['successful'] > 0:
            results['statistics'] = {
                'avg_preprocessing_ms': round(sum(results['metrics']['preprocessing_times_ms']) / results['successful'], 2),
                'avg_inference_ms': round(sum(results['metrics']['inference_times_ms']) / results['successful'], 2),
                'avg_total_ms': round(sum(results['metrics']['total_times_ms']) / results['successful'], 2),
                'min_inference_ms': round(min(results['metrics']['inference_times_ms']), 2),
                'max_inference_ms': round(max(results['metrics']['inference_times_ms']), 2),
                'throughput_images_per_sec': round(results['successful'] / (sum(results['metrics']['total_times_ms']) / 1000), 2)
            }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':
    load_model()
    
    print("\n" + "="*60)
    print("Swin V2 Grape Disease Detection API")
    print("="*60)
    print(f"API running on http://localhost:5000")
    print(f"Model: Swin V2 Transformer")
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    print(f"\nEndpoints:")
    print(f"  - GET  /health    - Health check")
    print(f"  - POST /predict   - Image classification")
    print(f"  - GET  /classes   - List available classes")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
