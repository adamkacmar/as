import requests
import time
import statistics
from pathlib import Path

API_URL = "http://localhost:5000"

def single_image_benchmark(image_path, num_runs=5):
    print(f"\n{'='*70}")
    print(f"Single Image Benchmark: {Path(image_path).name}")
    print(f"{'='*70}")
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    inference_times = []
    total_times = []
    
    print(f"\nRunning {num_runs} inference passes...")
    
    for i in range(num_runs):
        with open(image_path, 'rb') as img:
            files = {'image': img}
            response = requests.post(f"{API_URL}/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            metrics = result['metrics']
            inference_times.append(metrics['inference_time_ms'])
            total_times.append(metrics['total_time_ms'])
            
            print(f"  Run {i+1}: Inference={metrics['inference_time_ms']:.2f}ms, "
                  f"Total={metrics['total_time_ms']:.2f}ms")
        else:
            print(f"  Run {i+1}: Failed")
    
    if inference_times:
        print(f"\nInference Time Statistics:")
        print(f"  - Mean: {statistics.mean(inference_times):.2f} ms")
        print(f"  - Median: {statistics.median(inference_times):.2f} ms")
        print(f"  - Min: {min(inference_times):.2f} ms")
        print(f"  - Max: {max(inference_times):.2f} ms")
        if len(inference_times) > 1:
            print(f"  - StdDev: {statistics.stdev(inference_times):.2f} ms")
        print(f"  - Throughput: {1000 / statistics.mean(inference_times):.2f} img/sec")


def batch_benchmark(image_paths, description="Batch"):
    """
    Benchmark multiple images in one batch request
    
    Args:
        image_paths: List of image file paths
        description: Description of the batch
    """
    print(f"\n{'='*70}")
    print(f"Batch Benchmark: {description}")
    print(f"{'='*70}")
    
    valid_paths = [p for p in image_paths if Path(p).exists()]
    
    if not valid_paths:
        print(f"Error: No valid image files found")
        return
    
    print(f"\nBenchmarking {len(valid_paths)} images...")
    
    files = [('images', open(p, 'rb')) for p in valid_paths]
    
    try:
        response = requests.post(f"{API_URL}/benchmark", files=files)
        
        for _, f in files:
            f.close()
        
        if response.status_code == 200:
            result = response.json()
            stats = result.get('statistics', {})
            
            print(f"\nResults:")
            print(f"  - Total images: {result['total_images']}")
            print(f"  - Successful: {result['successful']}")
            print(f"  - Failed: {result['failed']}")
            
            if stats:
                print(f"\nAggregated Statistics:")
                print(f"  - Avg Preprocessing: {stats['avg_preprocessing_ms']:.2f} ms")
                print(f"  - Avg Inference: {stats['avg_inference_ms']:.2f} ms")
                print(f"  - Avg Total: {stats['avg_total_ms']:.2f} ms")
                print(f"  - Min Inference: {stats['min_inference_ms']:.2f} ms")
                print(f"  - Max Inference: {stats['max_inference_ms']:.2f} ms")
                print(f"  - Throughput: {stats['throughput_images_per_sec']:.2f} img/sec")
                print(f"  - Device: {result['device']}")
        else:
            print(f"Error: {response.json()}")
    
    except Exception as e:
            print(f"Error: {str(e)}")


def compare_performance(image_paths, num_single_runs=3):
    """
    Compare single-image vs batch processing performance
    
    Args:
        image_paths: List of image file paths
        num_single_runs: Number of runs for single-image benchmark
    """
    print(f"\n{'='*70}")
    print(f"Performance Comparison")
    print(f"{'='*70}")
    
    if not image_paths or not Path(image_paths[0]).exists():
        print("Error: No valid image files for comparison")
        return
    
    print("\nSingle Image Processing (sequential):")
    single_image_benchmark(image_paths[0], num_single_runs)
    
    print(f"\nBatch Processing ({len(image_paths)} images):")
    batch_benchmark(image_paths, "Comparison Batch")
    
    print(f"\n{'='*70}")
    print("Analysis:")
    print("  - Batch processing is more efficient for multiple images")
    print("  - Single image latency is important for real-time applications")
    print("  - Compare throughput and latency for your use case")
    print(f"{'='*70}\n")


def main():
    print("\n" + "Swin V2 API Benchmarking Examples")
    print("="*70)
    
    test_images = [
        "testlist.jpg",
    ]
    
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code != 200:
            print("Error: API is not running or not healthy")
            print(f"   Start the API with: python swinv2_api.py")
            return
    except Exception as e:
        print(f"Error: Cannot connect to API: {str(e)}")
        print(f"   Make sure the API is running on {API_URL}")
        return
    
    print("API is running and healthy\n")
    
    valid_images = [p for p in test_images if Path(p).exists()]
    
    if not valid_images:
        print("No test images found in current directory")
        print("Available test image paths:")
        for img in test_images:
            print(f"   - {img}")
        print("To run benchmarks, please:")
        print("1. Place your test images in the current directory")
        print("2. Update the 'test_images' list in this script")
        return
    
    print(f"Found {len(valid_images)} test image(s)\n")
    
    single_image_benchmark(valid_images[0], num_runs=5)
    
    if len(valid_images) > 1:
        batch_benchmark(valid_images, "Multiple Images")
    
    if len(valid_images) >= 2:
        compare_performance(valid_images, num_single_runs=3)
    
    print("\nBenchmarking complete!")


if __name__ == "__main__":
    main()
