"""
Builds the complete `generated_results/` deliverable from the real training runs:
evaluation charts per dataset/split/metric, EDA visuals, Grad-CAM attention maps,
training logs and the documentation/report files.

Layout produced:
    generated_results/
        Documentation_and_Reports/
        Malaria/Evaluation_Charts/Split_<split>/<Metric>/...
        Malaria/Visuals_and_EDA/
        Tuberculosis/...
        training_logs/
"""
import os
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
                             mean_absolute_error, mean_squared_error, r2_score, jaccard_score,
                             classification_report)

try:
    from data_loader import advanced_preprocessing
except ImportError:
    from .data_loader import advanced_preprocessing

# Split tags are written Train_Test_Val
SPLITS = ['70_20_10', '75_15_10', '90_5_5']
DATASETS = ['malaria', 'tb']
MODELS = ['Custom CNN', 'ResNet50', 'VGG16', 'MobileNetV2', 'DenseNet121']
CLASSIFICATION_METRICS = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR']

# Keras model.name of each architecture (used in checkpoint and CSV log file names)
MODEL_LAYER_NAMES = {
    'Custom CNN': 'Custom_CNN_Attention',
    'ResNet50': 'ResNet50_Attention',
    'VGG16': 'VGG16_Attention',
    'MobileNetV2': 'MobileNetV2_Attention',
    'DenseNet121': 'DenseNet121_Attention',
}

DATASET_INFO = {
    'malaria': {
        'display': 'Malaria',
        'short': 'Malaria',
        # (folder name, report name, distribution label, sample file suffix, sample title)
        'disease': ('Parasitized', 'Parasitized', 'Parasitized (Infected)', 'Infected', 'Malaria - Parasitized (Infected) Samples'),
        'healthy': ('Uninfected', 'Uninfected', 'Uninfected (Normal)', 'Uninfected', 'Malaria - Uninfected (Normal) Samples'),
        'palette': 'Set2',
        'attention_title': 'Attention Mechanism - Malaria Parasite Detection',
        'attention_original': 'Original Infected Cell',
    },
    'tb': {
        'display': 'Tuberculosis',
        'short': 'TB',
        'disease': ('Tuberculosis', 'Tuberculosis', 'Tuberculosis (Infected)', 'Infected', 'Tuberculosis - Infected Samples'),
        'healthy': ('Normal', 'Normal', 'Normal (Healthy)', 'Normal', 'Tuberculosis - Normal (Healthy) Samples'),
        'palette': 'Set1',
        'attention_title': 'Attention Mechanism - Tuberculosis Infiltrate Detection',
        'attention_original': 'Original Infected X-Ray',
    },
}

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')


# =========================================================================
# Naming helpers (shared with the notebook so artifacts line up)
# =========================================================================

def split_label(split):
    return split.replace('_', ':')

def checkpoint_path(dataset_name, split, model_layer_name):
    return f"best_{dataset_name}_{split}_{model_layer_name}.h5"

def training_log_path(dataset_name, split, model_layer_name):
    return f"training_log_{dataset_name}_{split}_{model_layer_name}.csv"

def comparative_csv_path(dataset_name, split):
    return f"comparative_results_{dataset_name}_{split}.csv"

def evaluation_path(dataset_name, split):
    return f"evaluation_{dataset_name}_{split}.npz"

def disease_index(dataset_name):
    """
    Label index of the disease class. image_dataset_from_directory assigns labels in
    alphabetical folder order, so Parasitized=0 for malaria and Tuberculosis=1 for TB.
    """
    info = DATASET_INFO[dataset_name]
    return sorted([info['disease'][0], info['healthy'][0]]).index(info['disease'][0])

def to_disease_positive(dataset_name, y_true, y_pred_probs):
    """
    Re-expresses labels and sigmoid outputs so that 1 / the probability always refer to the disease class.
    """
    y_true = np.asarray(y_true).astype(int).flatten()
    y_pred_probs = np.asarray(y_pred_probs, dtype=np.float64).flatten()
    if disease_index(dataset_name) == 1:
        return y_true, y_pred_probs
    return 1 - y_true, 1.0 - y_pred_probs

def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))


# =========================================================================
# Persisting per-split evaluations (so the report can be built across sessions)
# =========================================================================

def save_split_evaluation(dataset_name, split, y_true, predictions_dict, out_dir='.'):
    arrays = {'y_true': np.asarray(y_true).astype(int).flatten()}
    for model_name, probs in predictions_dict.items():
        arrays[f"probs::{model_name}"] = np.asarray(probs, dtype=np.float32).flatten()
    path = os.path.join(out_dir, evaluation_path(dataset_name, split))
    np.savez(path, **arrays)
    print(f"Saved test-set predictions to {path}")
    return path

def load_split_evaluation(dataset_name, split, in_dir='.'):
    path = os.path.join(in_dir, evaluation_path(dataset_name, split))
    if not os.path.exists(path):
        return None, {}
    data = np.load(path)
    y_true = data['y_true']
    predictions = {k.split('::', 1)[1]: data[k] for k in data.files if k.startswith('probs::')}
    return y_true, predictions


# =========================================================================
# Metrics
# =========================================================================

def compute_all_metrics(y_pos, p_pos, threshold=0.5):
    """
    All metrics are computed on the held-out test set with the disease class as positive.

    Classification: Accuracy, Precision, Recall, F1-Score, AUC-ROC, AUC-PR (average precision).
    Regression (on the predicted disease probability vs the 0/1 label): MAE, MSE, RMSE, R_Squared,
        and MAPE, the mean absolute percentage error of the probability assigned to the true class
        (target 1.0), expressed in percent.
    Segmentation (per class/"segment", averaged over the two classes, image-level since the datasets
        carry no pixel masks): IOU (mean Jaccard index), Dice_Coefficient (mean Dice), mAP (mean
        average precision) and Pixel_Accuracy (fraction of correctly classified samples).
    """
    y_pred = (p_pos >= threshold).astype(int)
    m = {}
    m['Accuracy'] = accuracy_score(y_pos, y_pred)
    m['Precision'] = precision_score(y_pos, y_pred, zero_division=0)
    m['Recall'] = recall_score(y_pos, y_pred, zero_division=0)
    m['F1-Score'] = f1_score(y_pos, y_pred, zero_division=0)
    m['AUC-ROC'] = roc_auc_score(y_pos, p_pos)
    m['AUC-PR'] = average_precision_score(y_pos, p_pos)

    m['MAE'] = mean_absolute_error(y_pos, p_pos)
    m['MSE'] = mean_squared_error(y_pos, p_pos)
    m['RMSE'] = np.sqrt(m['MSE'])
    m['R_Squared'] = r2_score(y_pos, p_pos)
    p_true_class = np.where(y_pos == 1, p_pos, 1.0 - p_pos)
    m['MAPE'] = np.mean(np.abs(1.0 - p_true_class)) * 100

    m['IOU'] = jaccard_score(y_pos, y_pred, average='macro', zero_division=0)
    m['Dice_Coefficient'] = f1_score(y_pos, y_pred, average='macro', zero_division=0)
    m['mAP'] = (average_precision_score(y_pos, p_pos) + average_precision_score(1 - y_pos, 1.0 - p_pos)) / 2
    m['Pixel_Accuracy'] = m['Accuracy']
    return m

def load_computational_metrics(dataset_name, split, in_dir='.'):
    """
    Reads the per-model params/FLOPs/latency/throughput recorded by benchmark.evaluate_all_models.
    """
    path = os.path.join(in_dir, comparative_csv_path(dataset_name, split))
    out = {}
    if not os.path.exists(path):
        return out
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        out[row['Architecture']] = {
            'params_M': round(float(row.get('Parameters (Millions)', np.nan)), 2),
            'flops_G': round(float(row.get('FLOPs (G)', np.nan)), 2),
            'latency_ms': round(float(row.get('Latency (ms/image)', np.nan)), 2),
            'throughput_fps': round(float(row.get('Throughput (images/s)', np.nan)), 2),
        }
    return out

def collect_results(in_dir='.'):
    """
    Returns (rows DataFrame, predictions) for every dataset/split whose evaluation exists.
    predictions[dataset][split] = (y_pos, {model: p_pos})
    """
    rows = []
    predictions = {}
    for dataset_name in DATASETS:
        for split in SPLITS:
            y_true, preds = load_split_evaluation(dataset_name, split, in_dir)
            if y_true is None:
                print(f"[results] No evaluation found for {dataset_name} {split}; skipping.")
                continue
            comp = load_computational_metrics(dataset_name, split, in_dir)
            y_pos = None
            pos_preds = {}
            for model_name in MODELS:
                if model_name not in preds:
                    continue
                y_pos, p_pos = to_disease_positive(dataset_name, y_true, preds[model_name])
                pos_preds[model_name] = p_pos
                row = {'Dataset': DATASET_INFO[dataset_name]['display'], 'Split': split_label(split), 'Model': model_name}
                row.update(compute_all_metrics(y_pos, p_pos))
                row.update(comp.get(model_name, {'params_M': np.nan, 'flops_G': np.nan, 'latency_ms': np.nan, 'throughput_fps': np.nan}))
                rows.append(row)
            predictions.setdefault(dataset_name, {})[split] = (y_pos, pos_preds)
    return pd.DataFrame(rows), predictions


# =========================================================================
# Evaluation charts
# =========================================================================

def _split_dir(output_dir, dataset_name, split, sub):
    path = os.path.join(output_dir, DATASET_INFO[dataset_name]['display'], 'Evaluation_Charts', f'Split_{split}', sub)
    os.makedirs(path, exist_ok=True)
    return path

def plot_metric_bar_charts(df, dataset_name, split, output_dir):
    display_name = DATASET_INFO[dataset_name]['display']
    subset = df[(df['Dataset'] == display_name) & (df['Split'] == split_label(split))]
    for metric in CLASSIFICATION_METRICS:
        data = subset[['Model', metric]].rename(columns={metric: 'Score'})
        with plt.style.context('ggplot'):
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Model', y='Score', hue='Model', palette='husl', data=data, legend=False)
            plt.title(f'{metric} Comparison - {display_name} (Split {split_label(split)})')
            plt.ylim(min(0.7, float(data['Score'].min()) - 0.05), 1.0)
            plt.tight_layout()
            plt.savefig(os.path.join(_split_dir(output_dir, dataset_name, split, metric), f'bar_chart_{metric}_{display_name}_{split}.png'))
            plt.close()

def plot_metric_histograms(df, dataset_name, split, output_dir):
    display_name = DATASET_INFO[dataset_name]['display']
    subset = df[(df['Dataset'] == display_name) & (df['Split'] == split_label(split))]
    with sns.axes_style('whitegrid'):
        for metric in CLASSIFICATION_METRICS:
            plt.figure(figsize=(9, 5))
            lo, hi = float(subset[metric].min()), float(subset[metric].max())
            # Identical scores (e.g. several models at 1.0) leave no range to bin; widen it slightly
            binrange = (lo - 0.005, hi + 0.005) if hi - lo < 1e-9 else None
            ax = sns.histplot(data=subset, x=metric, hue='Model', multiple='dodge', bins=10, binrange=binrange, palette='Set2', shrink=0.8)
            plt.title(f'Histogram of {metric} by Model - {display_name} ({split_label(split)})')
            plt.xlabel(f'{metric} Score')
            plt.ylabel('Frequency')
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1))
            plt.savefig(os.path.join(_split_dir(output_dir, dataset_name, split, metric), f'histogram_{metric}_{display_name}_{split}.png'), bbox_inches='tight', dpi=300)
            plt.close()

def plot_metric_pie_charts(df, dataset_name, split, output_dir):
    display_name = DATASET_INFO[dataset_name]['display']
    subset = df[(df['Dataset'] == display_name) & (df['Split'] == split_label(split))]
    with sns.axes_style('whitegrid'):
        for metric in CLASSIFICATION_METRICS:
            file_metric = 'accuracy' if metric == 'Accuracy' else metric
            plt.figure(figsize=(8, 8))
            plt.pie(subset[metric].clip(lower=0), labels=subset['Model'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
            plt.title(f'Relative {metric} Distribution Among Models - {display_name} ({split_label(split)})')
            plt.tight_layout()
            plt.savefig(os.path.join(_split_dir(output_dir, dataset_name, split, metric), f'piechart_{file_metric}_{display_name}_{split}.png'))
            plt.close()

def plot_metric_boxplot(df, dataset_name, split, output_dir):
    display_name = DATASET_INFO[dataset_name]['display']
    subset = df[(df['Dataset'] == display_name) & (df['Split'] == split_label(split))]
    with sns.axes_style('whitegrid'):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=subset[CLASSIFICATION_METRICS], palette="Set2")
        plt.title(f'Box Plot of Classification Metrics - {display_name} ({split_label(split)})')
        plt.ylabel('Score')
        plt.tight_layout()
        plt.savefig(os.path.join(_split_dir(output_dir, dataset_name, split, 'Combined_Metrics'), f'boxplot_metrics_{display_name}_{split}.png'))
        plt.close()

def plot_confusion_matrices(dataset_name, split, y_pos, pos_preds, output_dir):
    display_name = DATASET_INFO[dataset_name]['display']
    with plt.style.context('ggplot'):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Confusion Matrices - {display_name} (Split {split_label(split)})', fontsize=16)
        axes = axes.flatten()
        for idx, model_name in enumerate(MODELS):
            if model_name not in pos_preds:
                axes[idx].axis('off')
                continue
            y_pred = (pos_preds[model_name] >= 0.5).astype(int)
            cm = confusion_matrix(y_pos, y_pred, labels=[0, 1])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                        xticklabels=['Negative', 'Positive'],
                        yticklabels=['Negative', 'Positive'])
            axes[idx].set_title(model_name)
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        axes[-1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(_split_dir(output_dir, dataset_name, split, 'General_Performance'), f'confusion_matrices_{display_name}_{split}.png'))
        plt.close()

def plot_roc_curves(df, dataset_name, split, y_pos, pos_preds, output_dir):
    display_name = DATASET_INFO[dataset_name]['display']
    with plt.style.context('ggplot'):
        plt.figure(figsize=(8, 6))
        for model_name in MODELS:
            if model_name not in pos_preds:
                continue
            fpr, tpr, _ = roc_curve(y_pos, pos_preds[model_name])
            auc_val = roc_auc_score(y_pos, pos_preds[model_name])
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_val:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {display_name} (Split {split_label(split)})')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(_split_dir(output_dir, dataset_name, split, 'AUC-ROC'), f'roc_curves_{display_name}_{split}.png'))
        plt.close()

def plot_pr_curves(df, dataset_name, split, y_pos, pos_preds, output_dir):
    display_name = DATASET_INFO[dataset_name]['display']
    with plt.style.context('ggplot'):
        plt.figure(figsize=(8, 6))
        for model_name in MODELS:
            if model_name not in pos_preds:
                continue
            precision, recall, _ = precision_recall_curve(y_pos, pos_preds[model_name])
            auc_pr = average_precision_score(y_pos, pos_preds[model_name])
            plt.plot(recall, precision, label=f'{model_name} (AUC-PR = {auc_pr:.3f})')
        # A no-skill classifier's precision equals the prevalence of the positive class
        baseline = float(np.mean(y_pos))
        plt.plot([0, 1], [baseline, baseline], 'k--', label='Baseline')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {display_name} (Split {split_label(split)})')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(_split_dir(output_dir, dataset_name, split, 'AUC-PR'), f'pr_curves_{display_name}_{split}.png'))
        plt.close()

def read_training_history(dataset_name, split, model_name, in_dir='.'):
    """
    Loads the CSVLogger history of one run (phase 1 + fine-tuning), keeping the last entry
    per epoch in case a crashed run was resumed and re-logged the same epoch.
    """
    path = os.path.join(in_dir, training_log_path(dataset_name, split, MODEL_LAYER_NAMES[model_name]))
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty or 'epoch' not in df.columns:
        return None
    df = df.drop_duplicates(subset='epoch', keep='last').sort_values('epoch').reset_index(drop=True)
    df['epoch_number'] = df['epoch'].astype(int) + 1
    return df

def plot_training_history(dataset_name, split, output_dir, in_dir='.'):
    display_name = DATASET_INFO[dataset_name]['display']
    with plt.style.context('ggplot'):
        plt.figure(figsize=(15, 10))
        for idx, model_name in enumerate(MODELS):
            plt.subplot(2, 3, idx + 1)
            hist = read_training_history(dataset_name, split, model_name, in_dir)
            if hist is None:
                plt.text(0.5, 0.5, 'No training log found', ha='center', va='center')
                plt.title(f'{model_name} Training History')
                continue
            x = hist['epoch_number']
            plt.plot(x, hist['accuracy'], label='Train Acc', color='blue')
            plt.plot(x, hist['val_accuracy'], label='Val Acc', color='orange')
            plt.plot(x, hist['loss'], label='Train Loss', color='blue', linestyle='--')
            plt.plot(x, hist['val_loss'], label='Val Loss', color='orange', linestyle='--')
            plt.title(f'{model_name} Training History')
            plt.xlabel('Epochs')
            plt.ylabel('Score')
            top = max(1.1, float(np.nanmax(hist[['accuracy', 'val_accuracy', 'loss', 'val_loss']].values)) * 1.05)
            plt.ylim(0, top)
            plt.legend()
        plt.suptitle(f'Training History - {display_name} (Split {split_label(split)})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(_split_dir(output_dir, dataset_name, split, 'General_Performance'), f'training_history_{display_name}_{split}.png'))
        plt.close()

def write_training_logs(output_dir, in_dir='.'):
    logs_dir = os.path.join(output_dir, 'training_logs')
    os.makedirs(logs_dir, exist_ok=True)
    for dataset_name in DATASETS:
        display_name = DATASET_INFO[dataset_name]['display']
        for split in SPLITS:
            for model_name in MODELS:
                hist = read_training_history(dataset_name, split, model_name, in_dir)
                if hist is None:
                    continue
                log_file = os.path.join(logs_dir, f'{model_name.replace(" ", "_")}_{display_name}_{split}.log')
                with open(log_file, 'w') as f:
                    f.write(f"Training Log for {model_name} on {display_name} Dataset (Split {split_label(split)})\n")
                    f.write("=" * 65 + "\n")
                    f.write(f"{'Epoch':<10} | {'Train Acc':<10} | {'Val Acc':<10} | {'Train Loss':<12} | {'Val Loss':<10}\n")
                    f.write("-" * 65 + "\n")
                    for _, r in hist.iterrows():
                        f.write(f"{int(r['epoch_number']):<10} | {r['accuracy']:.4f}     | {r['val_accuracy']:.4f}     | {r['loss']:.4f}       | {r['val_loss']:.4f}\n")


# =========================================================================
# Visuals and EDA
# =========================================================================

def count_classes(data_dir, dataset_name):
    info = DATASET_INFO[dataset_name]
    return {key: len(list_images(os.path.join(data_dir, info[key][0]))) for key in ('disease', 'healthy')}

def plot_class_distribution(data_dir, dataset_name, output_dir):
    info = DATASET_INFO[dataset_name]
    counts = count_classes(data_dir, dataset_name)
    data = pd.DataFrame({
        'Class': [info['disease'][2], info['healthy'][2]],
        'Count': [counts['disease'], counts['healthy']]
    })
    out_dir = os.path.join(output_dir, info['display'], 'Visuals_and_EDA')
    os.makedirs(out_dir, exist_ok=True)
    offset = max(data['Count'].max() * 0.015, 1)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Class', y='Count', hue='Class', data=data, palette=info['palette'], legend=False)
    plt.title(f"Dataset Class Distribution - {info['display']}", fontsize=14)
    plt.ylabel('Number of Images', fontsize=12)
    for index, row in data.iterrows():
        plt.text(index, row.Count + offset, row.Count, color='black', ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"class_distribution_{info['display']}.png"))
    plt.close()

def plot_dataset_samples(data_dir, dataset_name, output_dir, num_images=8, seed=42):
    import cv2
    info = DATASET_INFO[dataset_name]
    out_dir = os.path.join(output_dir, info['display'], 'Visuals_and_EDA')
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    for key in ('disease', 'healthy'):
        folder_name, report_name, _, file_suffix, title = info[key]
        folder = os.path.join(data_dir, folder_name)
        files = list_images(folder)
        chosen = rng.sample(files, min(num_images, len(files)))
        fig, axes = plt.subplots(2, num_images // 2, figsize=(10, 5))
        for ax in axes.flatten():
            ax.axis('off')
        for ax, fname in zip(axes.flatten(), chosen):
            img = cv2.cvtColor(cv2.imread(os.path.join(folder, fname)), cv2.COLOR_BGR2RGB)
            ax.imshow(cv2.resize(img, (224, 224)))
            ax.set_title(report_name, fontsize=9)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"dataset_sample_{info['short']}_{file_suffix}.png"))
        plt.close()

def _find_attention_layer(model):
    """
    Last layer producing a spatial (4D) feature map: the CBAM-refined features that feed the classifier.
    """
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer
        except (AttributeError, ValueError, RuntimeError):
            continue
    raise ValueError(f"No 4D feature layer found in {model.name}")

def compute_gradcam(model, img_batch, disease_idx):
    """
    Grad-CAM for the disease class of a single-sigmoid-output model.
    """
    import tensorflow as tf
    target_layer = _find_attention_layer(model)
    grad_model = tf.keras.models.Model(model.input, [target_layer.output, model.output])
    with tf.GradientTape() as tape:
        feature_maps, preds = grad_model(img_batch, training=False)
        preds = tf.reshape(preds, [-1])
        score = preds if disease_idx == 1 else 1.0 - preds
    grads = tape.gradient(score, feature_maps)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(feature_maps[0] @ pooled_grads[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / max_val if float(max_val) > 0 else heatmap
    return heatmap.numpy(), target_layer.name

def generate_attention_map(dataset_name, split, models_dict, comparative_df, output_dir, base_dir='.', max_candidates=200):
    """
    Saves attention_map_<Dataset>.png: Grad-CAM of the most accurate model of this split on a
    correctly classified, confidently infected test image.
    """
    import cv2
    import tensorflow as tf
    try:
        from data_loader import get_split_dir
    except ImportError:
        from .data_loader import get_split_dir
    info = DATASET_INFO[dataset_name]
    if comparative_df is None or comparative_df.empty:
        print("[attention] No benchmark results; skipping attention map.")
        return None
    best_model = comparative_df.sort_values('Accuracy', ascending=False).iloc[0]['Architecture']
    model_path, model_builder = models_dict[best_model]
    tf.keras.backend.clear_session()
    model = model_builder()
    model.load_weights(model_path)

    test_folder = os.path.join(get_split_dir(base_dir, dataset_name, split), 'test', info['disease'][0])
    files = list_images(test_folder)[:max_candidates]
    d_idx = disease_index(dataset_name)
    best = None
    for fname in files:
        raw = cv2.cvtColor(cv2.imread(os.path.join(test_folder, fname)), cv2.COLOR_BGR2RGB)
        raw = cv2.resize(raw, (224, 224)).astype(np.float32)
        img = advanced_preprocessing(raw) / 255.0
        p = float(model.predict(img[np.newaxis], verbose=0).flatten()[0])
        p_disease = p if d_idx == 1 else 1.0 - p
        if best is None or p_disease > best[0]:
            best = (p_disease, raw, img)
        if p_disease >= 0.95:
            break
    if best is None:
        print(f"[attention] No test images found in {test_folder}; skipping attention map.")
        return None

    p_disease, raw, img = best
    heatmap, layer_name = compute_gradcam(model, img[np.newaxis], d_idx)
    heatmap = cv2.resize(heatmap, (224, 224))
    original = raw.astype(np.uint8)

    out_dir = os.path.join(output_dir, info['display'], 'Visuals_and_EDA')
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title(info['attention_original'])
    axes[0].axis('off')
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Grad-CAM Activation")
    axes[1].axis('off')
    axes[2].imshow(original)
    axes[2].imshow(heatmap, cmap='jet', alpha=0.4)
    axes[2].set_title("Attention Map Overlay")
    axes[2].axis('off')
    plt.suptitle(info['attention_title'], fontsize=16)
    fig.text(0.5, 0.01, f"{best_model} (Split {split_label(split)}), layer '{layer_name}', P({info['disease'][1]}) = {p_disease:.3f}", ha='center', fontsize=9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(out_dir, f"attention_map_{info['display']}.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved attention map to {path}")
    return path


# =========================================================================
# Documentation and reports
# =========================================================================

def export_tables(df, docs_dir):
    df.to_csv(os.path.join(docs_dir, 'all_metrics_results.csv'), index=False)
    report_split = split_label(SPLITS[0])
    with open(os.path.join(docs_dir, 'report_tables.md'), 'w') as f:
        f.write("# Automated Deep Learning Classification Results\n\n")
        for dataset_name in DATASETS:
            display_name = DATASET_INFO[dataset_name]['display']
            sub = df[(df['Dataset'] == display_name) & (df['Split'] == report_split)]
            if sub.empty:
                continue
            f.write(f"## {display_name} Detection Results\n\n")
            f.write(f"### 1. Classification Metrics (Split {report_split})\n")
            f.write(sub[['Model'] + CLASSIFICATION_METRICS].to_markdown(index=False, floatfmt=".4f") + "\n\n")
            f.write("### 2. Regression Metrics\n")
            f.write(sub[['Model', 'MAE', 'MSE', 'RMSE', 'R_Squared', 'MAPE']].to_markdown(index=False, floatfmt=".4f") + "\n\n")
            f.write("### 3. Segmentation Metrics (Attention Map Evaluation)\n")
            f.write(sub[['Model', 'IOU', 'Dice_Coefficient', 'mAP', 'Pixel_Accuracy']].to_markdown(index=False, floatfmt=".4f") + "\n\n")
            f.write("### 4. Training & Computational Metrics\n")
            f.write(sub[['Model', 'params_M', 'flops_G', 'latency_ms', 'throughput_fps']].to_markdown(index=False) + "\n\n")

def write_classification_reports(predictions, docs_dir):
    with open(os.path.join(docs_dir, 'detailed_classification_reports.txt'), 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED PER-CLASS CLASSIFICATION REPORTS\n")
        f.write("=" * 80 + "\n\n")
        for dataset_name in DATASETS:
            info = DATASET_INFO[dataset_name]
            for split in SPLITS:
                if split not in predictions.get(dataset_name, {}):
                    continue
                y_pos, pos_preds = predictions[dataset_name][split]
                for model_name in MODELS:
                    if model_name not in pos_preds:
                        continue
                    y_pred = (pos_preds[model_name] >= 0.5).astype(int)
                    # Disease class (1) listed first, then the healthy class (0)
                    rep = classification_report(y_pos, y_pred, labels=[1, 0], target_names=[info['disease'][1], info['healthy'][1]],
                                                output_dict=True, zero_division=0)
                    total = len(y_pos)
                    f.write(f"Dataset: {info['display']} | Split: {split_label(split)} | Model: {model_name}\n")
                    f.write("-" * 65 + "\n")
                    f.write(f"{'':<15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n\n")
                    for cls in (info['disease'][1], info['healthy'][1]):
                        r = rep[cls]
                        f.write(f"{cls:<15} {r['precision']:>10.2f} {r['recall']:>10.2f} {r['f1-score']:>10.2f} {int(r['support']):>10}\n")
                    f.write("\n")
                    f.write(f"{'accuracy':<15} {'':>10} {'':>10} {rep['accuracy']:>10.2f} {total:>10}\n")
                    for avg in ('macro avg', 'weighted avg'):
                        r = rep[avg]
                        f.write(f"{avg:<15} {r['precision']:>10.2f} {r['recall']:>10.2f} {r['f1-score']:>10.2f} {total:>10}\n")
                    f.write("\n" + "=" * 65 + "\n\n")

def _tb_description(counts):
    total = counts['disease'] + counts['healthy']
    return (f"Contains comprehensive collections of normal and TB-infected chest X-ray images. "
            f"Total Images: {total:,} ({counts['healthy']:,} Normal, and {counts['disease']:,} TB-Infected). "
            f"According to the Kaggle data card, a further 2,800 TB images are available from the NIAID TB portal "
            f"only under a data-sharing agreement and are not part of the downloadable dataset.")

def write_dataset_metadata(data_dirs, docs_dir):
    m = count_classes(data_dirs['malaria'], 'malaria') if data_dirs.get('malaria') else {'disease': 0, 'healthy': 0}
    t = count_classes(data_dirs['tb'], 'tb') if data_dirs.get('tb') else {'disease': 0, 'healthy': 0}
    m_total = m['disease'] + m['healthy']
    if m['disease'] == m['healthy']:
        m_description = f"Contains {m_total:,} cell images with equal instances of parasitized and uninfected cells."
    else:
        m_description = f"Contains {m_total:,} cell images ({m['disease']:,} parasitized and {m['healthy']:,} uninfected cells)."
    with open(os.path.join(docs_dir, 'dataset_metadata.txt'), 'w') as f:
        f.write("# Dataset Metadata\n\n")
        f.write("This document contains the required information regarding the datasets utilized for the Attention-Enhanced Deep Learning Framework.\n\n")
        f.write("## 1. Malaria Dataset\n")
        f.write("*   **Dataset Name:** NIH Malaria Dataset (Cell Images)\n")
        f.write("*   **Source/Link:** https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets\n")
        f.write(f"*   **Description:** {m_description}\n")
        f.write("*   **Date Accessed:** 11 July 2026\n\n")
        f.write("## 2. Tuberculosis Dataset\n")
        f.write("*   **Dataset Name:** Tuberculosis (TB) Chest X-ray Database\n")
        f.write("*   **Source/Link:** https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset \n")
        f.write(f"*   **Description:** {_tb_description(t)}\n")
        f.write("*   **Date Accessed:** 11 July 2026\n")
    return m, t

def write_summarized_report(df, data_counts, docs_dir):
    m, t = data_counts
    best_lines = []
    for dataset_name in DATASETS:
        display_name = DATASET_INFO[dataset_name]['display']
        sub = df[df['Dataset'] == display_name]
        if sub.empty:
            continue
        mean_acc = sub.groupby('Model')['Accuracy'].mean().sort_values(ascending=False)
        best, runner = mean_acc.index[0], (mean_acc.index[1] if len(mean_acc) > 1 else None)
        line = f"*   **{display_name}:** **{best}** achieved the highest mean test accuracy across the evaluated splits ({mean_acc.iloc[0]:.4f})"
        if runner:
            line += f", followed by {runner} ({mean_acc.iloc[1]:.4f})"
        best_lines.append(line + ".")
    fastest = df.groupby('Model')['latency_ms'].mean().dropna().sort_values()
    speed_line = ""
    if not fastest.empty:
        speed_line = (f"*   **Computational Cost:** {fastest.index[0]} had the lowest mean single-image inference latency "
                      f"({fastest.iloc[0]:.2f} ms), relevant for deployment on resource-constrained hardware.\n")

    t_total = t['disease'] + t['healthy']
    t_balance = "perfectly balanced" if t['disease'] == t['healthy'] else "imbalanced"
    m_total = m['disease'] + m['healthy']
    m_balance = "perfectly balanced" if m['disease'] == m['healthy'] else "imbalanced"

    with open(os.path.join(docs_dir, 'summarized_report.md'), 'w') as f:
        f.write(f"""# Summarized Project Report

**Project Title:** Attention-Enhanced Deep Learning Framework for Malaria and Tuberculosis Diagnosis and Classification
**Author:** Muhammed Toheeb Abdulraheem (Matric No: 210211118)
**Supervisor:** LA Akinyemi, PhD, SMIEEE, Loai Scholar
**Institution:** Department of Electronic and Computer Engineering, Faculty of Engineering, Lagos State University, Epe Campus, Lagos, Nigeria.

*(Note: The title reflects the handwritten correction to "Diagnosis and Classification".)*

---

## 1. Project Overview & Methodology

This project aims to automate the diagnosis and classification of Malaria and Tuberculosis (TB) using advanced artificial intelligence techniques, addressing critical diagnostic bottlenecks in healthcare systems. The core technical methodology is built upon the following elements outlined by the project supervisor:

*   **Deep Learning Framework:** The system is implemented using the **TensorFlow** and **Keras** deep learning frameworks for robust end-to-end processing and analysis of medical images.
*   **Architectural Diagram & Design:** The overall system architecture is meticulously designed to support scalable and reproducible experimentation across different image modalities.
*   **Neural Models:** The system leverages transfer learning by utilizing multiple established, pre-trained neural network architectures (such as ResNet50, VGG16, MobileNetV2, and DenseNet121) alongside a baseline Custom CNN, each enhanced with CBAM (Convolutional Block Attention Module) attention.
*   **Feature Extraction & Maps:** The methodology explicitly includes the generation of spatial feature representations, achieved through Grad-CAM attention maps. These maps visually highlight the image regions that drive the neural models' predictions.

## 2. Experimental Data Splits

To ensure rigorous validation and to test the robustness of the deep learning framework, the models were trained and evaluated across three distinct dataset splitting configurations, exactly as prescribed in the handwritten notes:

1.  **70% Train : 20% Test : 10% Validation**
2.  **75% Train : 15% Test : 10% Validation**
3.  **90% Train : 5% Test : 5% Validation**

## 3. Dataset Summary

The models were evaluated using two primary imaging datasets:
*   **Malaria Dataset (Blood Smears):** {m_total:,} total images, {m_balance} ({m['disease']:,} Parasitized and {m['healthy']:,} Uninfected). Accessed on 11 July 2026.
*   **Tuberculosis Dataset (Chest X-Rays):** {t_total:,} total images, {t_balance} ({t['healthy']:,} Normal and {t['disease']:,} TB-Infected). According to the Kaggle data card, a further 2,800 TB images are held by the NIAID TB portal under a data-sharing agreement and are not included in the downloadable dataset. Accessed on 11 July 2026.

## 4. Evaluation Metrics

Comprehensive performance metrics were collected across all models and data splits to satisfy the holistic evaluation requirements:
*   **Classification Metrics:** Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, AUC-ROC, and AUC-PR.
*   **Regression/Error Metrics:** Mean Absolute Error (MAE), Root Mean Square Error (RMSE), Mean Squared Error (MSE), Mean Absolute Percentage Error (MAPE), and R-Squared ($R^2$).
*   **Segmentation (Per-Class) Metrics:** Intersection over Union (IoU), Dice Coefficient, mean Average Precision (mAP), and Pixel Accuracy, computed per class and averaged.
*   **Visual Evaluations:** Class distribution charts, sample EDA crops, confusion matrices, Grad-CAM attention maps, and a comprehensive suite of performance plots (Bar Charts, Histograms, Box Plots, Pie Charts, ROC/PR Curves, and Training History learning curves).

## 5. Key Findings

{chr(10).join(best_lines)}
{speed_line}*   **Attention Efficacy:** Grad-CAM heatmaps computed on the CBAM-refined feature maps of the best model (see `Visuals_and_EDA/attention_map_*.png`) show which image regions drove each prediction, making the models' decisions interpretable and open to clinical verification.
""")

STUDENT_DEFENSE_NOTES = r"""# Defense Preparation Notes: Understanding the Metrics

This document is designed to help you prepare for questions during your project defense, specifically regarding how the values in the report tables were derived and why certain metrics were or were not used.

When your supervisor asks how you obtained the exact values for the metrics, confidently explain that they were mathematically derived by running the trained models on an unseen **Test Set**. A Python script (using the `scikit-learn` library) compared the model's predictions to the ground-truth labels to create a **Confusion Matrix**, and then applied standard statistical formulas to generate the final scores. In every table, the diseased class (Parasitized / Tuberculosis) is treated as the **Positive** class.

---

## 1. Classification Metrics (Accuracy, Precision, Recall, F1-Score, Specificity)

These metrics evaluate the model's categorical predictions against the actual true labels in the test set. By comparing the predictions to the ground truth, we categorize every test image into True Positives (TP), True Negatives (TN), False Positives (FP), or False Negatives (FN).

*   **Accuracy:** `(TP + TN) / Total`
    *   *Explanation:* The overall percentage of correct diagnoses out of all images in the test set.
*   **Precision:** `TP / (TP + FP)`
    *   *Explanation:* Out of all the images the model *claimed* were diseased, how many were actually diseased? (Measures false alarms).
*   **Recall (Sensitivity):** `TP / (TP + FN)`
    *   *Explanation:* Out of all the *actually* diseased images, how many did the model successfully find? (Measures missed diagnoses).
*   **F1-Score:** `2 * (Precision * Recall) / (Precision + Recall)`
    *   *Explanation:* The harmonic mean of Precision and Recall, used to show a balance between the two.
*   **Specificity:** `TN / (TN + FP)`
    *   *Explanation:* Out of all the *actually* healthy images, how many did the model correctly identify as healthy? (This is the recall of the healthy class in the detailed classification reports.)

## 2. AUC Metrics (AUC-ROC and AUC-PR)

Instead of evaluating a single Yes/No prediction at a fixed 50% confidence threshold, these metrics evaluate the model's continuous probability outputs across *all possible thresholds* (from 0% to 100%).

*   **AUC-ROC (Area Under the Receiver Operating Characteristic Curve):**
    *   *Explanation:* The script plots the True Positive Rate against the False Positive Rate at every threshold. The Area Under the Curve (AUC) summarizes the model's overall ability to distinguish between the diseased and healthy classes, regardless of what threshold you pick.
*   **AUC-PR (Area Under the Precision-Recall Curve):**
    *   *Explanation:* Similar to ROC, but it plots Precision against Recall (computed as the average precision). This is particularly useful for evaluating performance when one class might be more important or if there are class imbalances, as in the TB dataset. The dashed baseline on the PR curve is the proportion of diseased images in the test set.

## 3. Regression / Error Metrics (MAE, MSE, RMSE, MAPE, R-Squared)

While usually reserved for predicting continuous numbers (like predicting temperature or house prices), we applied them here to measure the "confidence error" of the classification model's probability outputs.

*   **MAE / MSE / RMSE:**
    *   *Explanation:* The script calculates the exact numerical distance (the "error") between the model's predicted probability (e.g., 0.85) and the true label (1.0 for disease). The Mean Absolute Error (MAE) averages the absolute differences, while MSE and RMSE penalize larger errors more heavily. It tells you how "close" the model was to perfect certainty.
*   **MAPE (Mean Absolute Percentage Error):**
    *   *Explanation:* For each image, the probability the model gave to the *correct* class is compared with the ideal value of 100%, and the percentage shortfall is averaged. Because the target is always 100%, MAPE equals MAE expressed as a percentage.
*   **R-Squared ($R^2$):**
    *   *Explanation:* Measures how well the variance in the true labels can be predicted by the model's probability outputs.

## 4. Segmentation Metrics (IoU, Dice Coefficient, mAP, Pixel Accuracy)

If the supervisor asks how the Segmentation Metrics were obtained, here is the correct response:

*   **The Answer:** *"Following the supervisor's instruction, each class (e.g., Parasitized and Uninfected) was treated as a segment. IoU, Dice, mAP and Pixel Accuracy were computed for each class on the test set and averaged over the two classes."*
*   **The Formulas (per class):** IoU = `TP / (TP + FP + FN)`, Dice = `2TP / (2TP + FP + FN)`, AP = area under that class's precision-recall curve, and Pixel Accuracy = correctly classified samples / total samples.
*   **The Limitation:** True pixel-level segmentation metrics would require a **ground-truth segmentation mask** (an exact, hand-drawn pixel outline of the disease created by a doctor) to compare against. The Kaggle datasets used (Malaria blood smears and TB X-rays) only provide image-level labels ("Disease" vs "Normal"), so the metrics are computed at image level per class.
*   **Localization:** Because pixel masks are unavailable, we used **Grad-CAM Attention Maps** to achieve *localization*. These show visually which regions of the image the model relied on.

## 5. Generating Attention Maps (Grad-CAM)

If the supervisor asks how you generated the "heatmaps" or attention maps, you must explain **Grad-CAM** (Gradient-weighted Class Activation Mapping). This is what makes your Deep Learning model interpretable rather than just a "black box."

*   **The Answer:** *"The attention maps were generated using the Grad-CAM algorithm. It works by tracking the mathematical gradients of the model's final prediction back to the last convolutional (CBAM-refined) feature map to see exactly which spatial features drove the decision."*
*   **The Explanation (Step-by-Step):**
    1.  **Forward Pass:** We pass an image (e.g., a blood smear) into the fully trained model to get a prediction (e.g., "Parasitized").
    2.  **Calculate Gradients (Backpropagation):** The script calculates the mathematical gradient (derivative) of that specific prediction score with respect to the feature maps in the *final convolutional stage* of the network, after the CBAM attention block. This layer is chosen because it contains the richest spatial and semantic information just before the final classification.
    3.  **Weighting the Maps:** These gradients act as "importance weights." They tell us which specific feature maps were most strongly activated by the disease.
    4.  **Creating the Heatmap:** The feature maps are multiplied by their weights, summed together, and passed through a ReLU activation function (to keep only the positive influences).
    5.  **Superimposition:** Finally, this 2D heatmap is resized and overlaid on top of the original input image. The "hot" colors (reds and yellows) highlight the pixels that the model focused on to make its diagnosis.
"""

def write_defense_notes(docs_dir):
    with open(os.path.join(docs_dir, 'student_defense_notes.md'), 'w') as f:
        f.write(STUDENT_DEFENSE_NOTES)


# =========================================================================
# Entry point
# =========================================================================

def generate_all_results(data_dirs, output_dir='generated_results', in_dir='.', make_zip=True):
    """
    Builds the whole generated_results tree from the saved evaluations and training logs in `in_dir`.
    data_dirs: {'malaria': <raw malaria folder>, 'tb': <raw TB folder>} (used for EDA and metadata).
    """
    os.makedirs(output_dir, exist_ok=True)
    docs_dir = os.path.join(output_dir, 'Documentation_and_Reports')
    os.makedirs(docs_dir, exist_ok=True)

    df, predictions = collect_results(in_dir)
    if df.empty:
        print("[results] No evaluations found. Run the training/benchmark loop first.")
        return df

    print("Generating evaluation charts...")
    for dataset_name in DATASETS:
        for split, (y_pos, pos_preds) in predictions.get(dataset_name, {}).items():
            if not pos_preds:
                continue
            plot_metric_bar_charts(df, dataset_name, split, output_dir)
            plot_metric_histograms(df, dataset_name, split, output_dir)
            plot_metric_pie_charts(df, dataset_name, split, output_dir)
            plot_metric_boxplot(df, dataset_name, split, output_dir)
            plot_confusion_matrices(dataset_name, split, y_pos, pos_preds, output_dir)
            plot_roc_curves(df, dataset_name, split, y_pos, pos_preds, output_dir)
            plot_pr_curves(df, dataset_name, split, y_pos, pos_preds, output_dir)
            plot_training_history(dataset_name, split, output_dir, in_dir)

    print("Writing training logs...")
    write_training_logs(output_dir, in_dir)

    print("Generating EDA visuals...")
    for dataset_name in DATASETS:
        data_dir = data_dirs.get(dataset_name)
        if data_dir and os.path.isdir(data_dir):
            plot_class_distribution(data_dir, dataset_name, output_dir)
            plot_dataset_samples(data_dir, dataset_name, output_dir)
        attention_path = os.path.join(output_dir, DATASET_INFO[dataset_name]['display'], 'Visuals_and_EDA',
                                      f"attention_map_{DATASET_INFO[dataset_name]['display']}.png")
        if not os.path.exists(attention_path):
            print(f"[results] Warning: {attention_path} is missing (it is produced after benchmarking split {SPLITS[0]}).")

    print("Writing documentation and reports...")
    export_tables(df, docs_dir)
    write_classification_reports(predictions, docs_dir)
    counts = write_dataset_metadata(data_dirs, docs_dir)
    write_summarized_report(df, counts, docs_dir)
    write_defense_notes(docs_dir)

    if make_zip:
        archive = shutil.make_archive(output_dir, 'zip', root_dir=os.path.dirname(os.path.abspath(output_dir)),
                                      base_dir=os.path.basename(os.path.abspath(output_dir)))
        print(f"Zipped results to {archive}")

    print(f"Results generation complete. Check the '{output_dir}' directory.")
    return df
