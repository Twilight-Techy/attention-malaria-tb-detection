import time
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from utils import evaluate_comprehensive_metrics

def measure_inference_latency(model, test_data, num_samples=100):
    """
    Measures the average inference latency per image in milliseconds.
    """
    print(f"Measuring latency for {model.name}...")
    images, _ = next(iter(test_data))
    
    # Warm up (TF graph compilation)
    _ = model.predict(images[:1], verbose=0)
    
    start_time = time.time()
    for i in range(min(num_samples, len(images))):
        _ = model.predict(images[i:i+1], verbose=0)
    end_time = time.time()
    
    avg_latency_ms = ((end_time - start_time) / min(num_samples, len(images))) * 1000
    return avg_latency_ms

def get_model_size_mb(model_path):
    if os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    return 0.0

def evaluate_all_models(models_dict, test_data, dataset_name, output_csv="comparative_results.csv"):
    results = []
    predictions_dict = {}
    
    y_true = []
    all_images = []
    
    print("Extracting test data for benchmarking...")
    batches = len(test_data) 
    count = 0
    for img, labels in test_data:
        y_true.extend(labels)
        all_images.append(img)
        count += 1
        if count >= batches:
            break
            
    y_true = np.array(y_true)
    all_images = np.vstack(all_images)

    from attention import cbam_block, channel_attention, spatial_attention
    custom_objects = {'cbam_block': cbam_block, 'channel_attention': channel_attention, 'spatial_attention': spatial_attention}

    for model_name, model_path in models_dict.items():
        if not os.path.exists(model_path):
            print(f"Skipping {model_name}: Model file '{model_path}' not found.")
            continue
            
        print(f"\n--- Evaluating {model_name} ---")
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        
        param_count = model.count_params()
        size_mb = get_model_size_mb(model_path)
        latency_ms = measure_inference_latency(model, test_data)
        
        print("Generating predictions...")
        y_pred_probs = model.predict(all_images, verbose=0).flatten()
        predictions_dict[model_name] = y_pred_probs
        
        metrics = evaluate_comprehensive_metrics(y_true, y_pred_probs)
        
        row = {
            "Architecture": model_name,
            "Dataset": dataset_name,
            "Accuracy": metrics.get('accuracy', 0),
            "Sensitivity (Recall)": metrics.get('recall', 0),
            "Specificity": metrics.get('specificity', 0),
            "Precision": metrics.get('precision', 0),
            "F1-Score": metrics.get('f1', 0),
            "MAE": metrics.get('mae', 0),
            "RMSE": metrics.get('rmse', 0),
            "Parameters (Millions)": param_count / 1e6,
            "Size (MB)": size_mb,
            "Latency (ms/image)": latency_ms
        }
        results.append(row)
        
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nComparative results saved to {output_csv}")
    
    return df, y_true, predictions_dict
