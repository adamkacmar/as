import requests
import json
from pathlib import Path

API_URL = "http://localhost:5000"

def test_health():
    print("\n" + "="*60)
    print("Testing /health endpoint...")
    print("="*60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_get_classes():
    print("\n" + "="*60)
    print("Testing /classes endpoint...")
    print("="*60)
    
    response = requests.get(f"{API_URL}/classes")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_predict(image_path):
    print("\n" + "="*60)
    print(f"Testing /predict endpoint with image: {image_path}")
    print("="*60)
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        return False
    
    with open(image_path, 'rb') as img_file:
        files = {'image': img_file}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction Result:")
        print(f"  - Predicted Class: {result['predicted_class']}")
        print(f"  - Confidence: {result['confidence']:.4f}")
        
        print(f"\n  Top Predictions:")
        for i, pred in enumerate(result['top_predictions'], 1):
            print(f"    {i}. {pred['class']}: {pred['confidence']:.4f}")
        
        if 'metrics' in result:
            print(f"\n  Performance Metrics:")
            print(f"    - Preprocessing: {result['metrics']['preprocessing_time_ms']:.2f} ms")
            print(f"    - Inference: {result['metrics']['inference_time_ms']:.2f} ms")
            print(f"    - Total: {result['metrics']['total_time_ms']:.2f} ms")
            print(f"    - Device: {result['metrics']['device']}")
        
        return True
    else:
        print(f"Error: {response.json()}")
        return False


def test_benchmark():
    print("\n" + "="*60)
    print("Testing /benchmark endpoint...")
    print("="*60)
    
    test_image_path = "testlist.jpg"
    
    if not Path(test_image_path).exists():
        print(f"No test image found at '{test_image_path}'")
        print("To test benchmarking, create test images or update the path")
        return False
    
    images = []
    for i in range(100):
        if Path(test_image_path).exists():
            images.append(('images', open(test_image_path, 'rb')))
    
    if not images:
        print("No images available for benchmarking")
        return False
    
    try:
        response = requests.post(f"{API_URL}/benchmark", files=images)
        
        for _, f in images:
            f.close()
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nBenchmark Results:")
            print(f"  Total images: {result['total_images']}")
            print(f"  Successful: {result['successful']}")
            print(f"  Failed: {result['failed']}")
            
            if 'statistics' in result:
                stats = result['statistics']
                print(f"\n  Statistics:")
                print(f"    - Avg Preprocessing: {stats['avg_preprocessing_ms']:.2f} ms")
                print(f"    - Avg Inference: {stats['avg_inference_ms']:.2f} ms")
                print(f"    - Avg Total: {stats['avg_total_ms']:.2f} ms")
                print(f"    - Min Inference: {stats['min_inference_ms']:.2f} ms")
                print(f"    - Max Inference: {stats['max_inference_ms']:.2f} ms")
                print(f"    - Throughput: {stats['throughput_images_per_sec']:.2f} img/s")
                print(f"    - Device: {result['device']}")
            
            return True
        else:
            print(f"Error: {response.json()}")
            return False
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def main():
    print("\n" + "="*60)
    print("Swin V2 API Test Client")
    print("="*60)
    
    if not test_health():
        print("\nHealth check failed! Make sure the API is running.")
        return
    
    test_get_classes()
    
    test_image_path = "testlist.jpg"
    
    if Path(test_image_path).exists():
        test_predict(test_image_path)
        
        test_benchmark()
    else:
        print(f"\nNo test image found at '{test_image_path}'")
        print("To test predictions and benchmarking, create a test image or update the path in test_client.py")
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
