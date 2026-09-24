import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, roc_curve, auc
from statsmodels.stats.contingency_tables import mcnemar

def evaluate_comprehensive_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Calculates Accuracy, Precision, Recall, Specificity, F1-Score, MAE, and RMSE.
    """
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred_probs)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_probs))
    
    print("--- Comprehensive Evaluation Metrics ---")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"MAE:         {mae:.4f}")
    print(f"RMSE:        {rmse:.4f}")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'specificity': specificity, 'f1': f1, 'mae': mae, 'rmse': rmse}

def plot_training_history_from_csv(csv_path, title="Training History"):
    """
    Reads a CSV log file and plots the training and validation loss/accuracy.
    """
    import pandas as pd
    
    if not os.path.exists(csv_path):
        print(f"Log file {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(12, 4))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    if 'accuracy' in df.columns:
        plt.plot(df['epoch'], df['accuracy'], label='Train Accuracy', marker='o')
    if 'val_accuracy' in df.columns:
        plt.plot(df['epoch'], df['val_accuracy'], label='Val Accuracy', marker='o')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    if 'loss' in df.columns:
        plt.plot(df['epoch'], df['loss'], label='Train Loss', marker='o')
    if 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='o')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_comparative_roc(y_true, predictions_dict, title="Comparative ROC Curve", save_path=None):
    plt.figure(figsize=(10, 8))
    for model_name, y_pred_probs in predictions_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_comparative_bar_chart(df, metric='F1-Score', title=None, save_path=None):
    if df.empty: return
    plt.figure(figsize=(10, 6))
    models = df['Architecture']
    values = df[metric]
    bars = plt.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.01), f'{yval:.3f}', ha='center', va='bottom')
        
    plt.xlabel('Architecture')
    plt.ylabel(metric)
    plt.title(title if title else f'Comparative {metric}')
    plt.ylim(0, max(values) * 1.1)
    plt.xticks(rotation=15)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def perform_mcnemar_test(y_true, y_pred_model1, y_pred_model2, threshold=0.5):
    """
    Performs McNemar's statistical test to compare two models' predictions.
    Null hypothesis: the two models have the same error rate.
    """
    pred1 = (y_pred_model1 >= threshold).astype(int)
    pred2 = (y_pred_model2 >= threshold).astype(int)
    
    # Create contingency table
    both_correct = np.sum((pred1 == y_true) & (pred2 == y_true))
    m1_correct_m2_wrong = np.sum((pred1 == y_true) & (pred2 != y_true))
    m1_wrong_m2_correct = np.sum((pred1 != y_true) & (pred2 == y_true))
    both_wrong = np.sum((pred1 != y_true) & (pred2 != y_true))
    
    table = [[both_correct, m1_correct_m2_wrong],
             [m1_wrong_m2_correct, both_wrong]]
             
    # Perform the test
    result = mcnemar(table, exact=False, correction=True)
    
    print("--- McNemar's Test ---")
    print(f"Statistic: {result.statistic:.4f}")
    print(f"p-value:   {result.pvalue:.4e}")
    
    if result.pvalue < 0.05:
        print("Result: Significant difference between the two models (Reject Null Hypothesis)")
    else:
        print("Result: No significant difference between the two models (Fail to Reject Null Hypothesis)")
        
    return result

def plot_training_history(history, model_name="Model"):
    """
    Plots the training and validation accuracy and loss.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{model_name} Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{model_name} Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def get_img_array(img_path, size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array / 255.0

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a single-sigmoid-output model.
    last_conv_layer_name: layer to explain; defaults to the last spatial (CBAM-refined) feature map.
    pred_index: 1 explains the sigmoid output (label 1), 0 explains its complement (label 0);
                defaults to the predicted label.
    """
    if last_conv_layer_name is None:
        from results import _find_attention_layer
        target_layer = _find_attention_layer(model)
    else:
        target_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(model.input, [target_layer.output, model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array, training=False)
        preds = tf.reshape(preds, [-1])
        if pred_index is None:
            pred_index = int(float(preds[0]) >= 0.5)
        class_channel = preds if pred_index == 1 else 1.0 - preds

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    heatmap = heatmap / max_val if float(max_val) > 0 else heatmap
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.4):
    """
    Overlays the Grad-CAM heatmap on the original image.
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    
    heatmap = cv2.resize(np.uint8(255 * heatmap), (224, 224))
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = np.clip(jet * alpha + img, 0, 255)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM")
    plt.imshow(cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB))
    plt.show()

def _train_dir(dataset_name, split=None, base_dir='.'):
    from data_loader import get_split_dir
    key = "malaria" if dataset_name.lower() == "malaria" else "tb"
    return os.path.join(get_split_dir(os.path.abspath(base_dir), key, split), "train")

def plot_class_distribution(train_gen, dataset_name, save_path=None, split=None, base_dir='.'):
    """
    Plots a bar chart showing the balance of classes in the training set of a split.
    """
    train_dir = _train_dir(dataset_name, split, base_dir)
    if not os.path.exists(train_dir):
        print(f"Training folder {train_dir} not found; load the split first.")
        return

    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    values = [len(os.listdir(os.path.join(train_dir, d))) for d in classes]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(classes, values, color=['#1f77b4', '#ff7f0e'][:len(classes)])
    plt.title(f'{dataset_name.upper()} - Training Class Distribution')
    plt.ylabel('Number of Images')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.01), int(yval), ha='center', va='bottom')
        
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_sample_images(train_gen, dataset_name, num_images=16, save_path=None, split=None, base_dir='.'):
    """
    Plots a grid of sample images, guaranteeing an equal split between classes.
    """
    # Infer class names from dataset directory structure
    train_dir = _train_dir(dataset_name, split, base_dir)
        
    if os.path.exists(train_dir):
        class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        labels_dict = {i: name for i, name in enumerate(class_names)}
    else:
        labels_dict = {0: "Class 0", 1: "Class 1"}
    
    target_per_class = num_images // 2
    collected_images = {0: [], 1: []}
    
    # Safely iterate through the tf.data.Dataset
    for images, labels in train_gen:
        images_np = images.numpy() if hasattr(images, 'numpy') else images
        labels_np = labels.numpy() if hasattr(labels, 'numpy') else labels
        
        for i in range(len(images_np)):
            if len(labels_np.shape) > 1 and labels_np.shape[1] > 1:
                class_idx = np.argmax(labels_np[i])
            else:
                # Handle scalar extraction safely
                class_idx = int(labels_np[i].item() if hasattr(labels_np[i], 'item') else labels_np[i])
                
            if class_idx in collected_images and len(collected_images[class_idx]) < target_per_class:
                collected_images[class_idx].append(images_np[i])
                
        if len(collected_images[0]) >= target_per_class and len(collected_images[1]) >= target_per_class:
            break
            
    final_images = collected_images[0] + collected_images[1]
    final_labels = [0] * target_per_class + [1] * target_per_class
    
    plt.figure(figsize=(10, 10))
    plt.suptitle(f"Sample {dataset_name.upper()} Images", fontsize=16)
    
    grid_size = int(np.ceil(np.sqrt(num_images)))
    for i in range(num_images):
        plt.subplot(grid_size, grid_size, i + 1)
        # Ensure image is safely plotted whether it is [0, 1] or [0, 255]
        img_to_plot = final_images[i]
        if img_to_plot.max() > 1.0:
            img_to_plot = img_to_plot / 255.0
        plt.imshow(img_to_plot)
        plt.title(labels_dict.get(final_labels[i], "Unknown"))
        plt.axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
