import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, mean_squared_error
from statsmodels.stats.contingency_tables import mcnemar

def evaluate_comprehensive_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Calculates Accuracy, Precision, Recall, Specificity, F1-Score, MAE, and RMSE.
    """
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred_probs)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_probs))
    
    print("--- Comprehensive Evaluation Metrics ---")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"MAE:         {mae:.4f}")
    print(f"RMSE:        {rmse:.4f}")
    
    return {'specificity': specificity, 'f1': f1, 'mae': mae, 'rmse': rmse}

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
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
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

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates Grad-CAM heatmap for a given image array and model.
    """
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.4):
    """
    Overlays the Grad-CAM heatmap on the original image.
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = jet * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM")
    plt.imshow(superimposed_img)
    plt.show()
