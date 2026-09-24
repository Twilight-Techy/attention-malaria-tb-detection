import time
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from utils import evaluate_comprehensive_metrics
from results import to_disease_positive

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

def compute_flops_g(model, input_shape=(224, 224, 3)):
    """
    Counts the floating point operations of a single-image forward pass (in GFLOPs)
    by freezing the model graph and running the TensorFlow profiler over it.
    """
    try:
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
        forward = tf.function(lambda x: model(x, training=False))
        concrete = forward.get_concrete_function(tf.TensorSpec([1, *input_shape], tf.float32))
        frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete)
        options = tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        ).with_empty_output().build()
        info = tf.compat.v1.profiler.profile(graph=frozen_func.graph, run_meta=tf.compat.v1.RunMetadata(), cmd='op', options=options)
        return info.total_float_ops / 1e9
    except Exception as e:
        print(f"Could not compute FLOPs for {model.name}: {e}")
        return float('nan')

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
    # Report everything with the disease class (Parasitized / Tuberculosis) as the positive class
    y_true_pos, _ = to_disease_positive(dataset_name, y_true, np.zeros(len(y_true)))

    for model_name, (model_path, model_builder) in models_dict.items():
        if not os.path.exists(model_path):
            print(f"Skipping {model_name}: Model file '{model_path}' not found.")
            continue
            
        print(f"\n--- Evaluating {model_name} ---")
        model = model_builder()
        model.load_weights(model_path)
        
        param_count = model.count_params()
        size_mb = get_model_size_mb(model_path)
        latency_ms = measure_inference_latency(model, test_data)
        flops_g = compute_flops_g(model)
        
        print("Generating predictions...")
        # Warm up at the batch size used below so graph tracing is not timed
        _ = model.predict(all_images[:32], batch_size=32, verbose=0)
        start_time = time.time()
        y_pred_probs = model.predict(all_images, batch_size=32, verbose=0).flatten()
        throughput_fps = len(all_images) / (time.time() - start_time)
        _, y_pred_probs = to_disease_positive(dataset_name, y_true, y_pred_probs)
        predictions_dict[model_name] = y_pred_probs
        
        metrics = evaluate_comprehensive_metrics(y_true_pos, y_pred_probs)
        
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
            "Latency (ms/image)": latency_ms,
            "Throughput (images/s)": throughput_fps,
            "FLOPs (G)": flops_g
        }
        results.append(row)
        
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nComparative results saved to {output_csv}")
    
    return df, y_true_pos, predictions_dict
