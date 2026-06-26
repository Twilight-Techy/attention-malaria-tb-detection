import nbformat as nbf

nb = nbf.v4.new_notebook()

title_cell = nbf.v4.new_markdown_cell("""# Attention-Improved Deep Learning Framework for Malaria and Tuberculosis Detection and Classification
**Author:** Muhammed Toheeb Abdulraheem

**Abstract / Executive Summary:**
This computational notebook orchestrates the experimental pipeline for a thesis focused on developing an attention-improved deep learning framework. It systematically facilitates the acquisition, preprocessing, training, evaluation, and interpretability analysis of five distinct convolutional architectures (a Custom CNN, ResNet50, VGG16, MobileNetV2, and DenseNet121) enhanced with Convolutional Block Attention Modules (CBAM). The framework is designed to optimize the automatic classification of Malaria from thin blood smear microscopic images and Tuberculosis from chest X-ray radiographs.
""")

setup_cell = nbf.v4.new_code_cell("""import sys
import os
import matplotlib.pyplot as plt
import getpass

# 1. Install Dependencies
print("Installing dependencies...")
!pip install -q -r requirements.txt

# 2. Setup Kaggle Credentials for Dataset Download
kaggle_json_path = os.path.expanduser('~/.kaggle/kaggle.json')
if not os.path.exists(kaggle_json_path):
    print("Kaggle credentials not found.")
    if 'KAGGLE_USERNAME' not in os.environ or 'KAGGLE_KEY' not in os.environ:
        print("Please provide your Kaggle API credentials to download the datasets (from kaggle.com -> Account -> Create New API Token).")
        os.environ['KAGGLE_USERNAME'] = input("Kaggle Username: ")
        os.environ['KAGGLE_KEY'] = getpass.getpass("Kaggle Key: ")

# 3. Add src to path
sys.path.append(os.path.abspath('src'))

try:
    from data_loader import load_malaria_data, load_tb_data
    from models import build_custom_cnn_attention, build_resnet50_attention, build_vgg16_attention, build_mobilenetv2_attention, build_densenet121_attention
    from train import compile_model, train_model, unfreeze_and_finetune
    from utils import plot_training_history, make_gradcam_heatmap, display_gradcam, evaluate_comprehensive_metrics, perform_mcnemar_test, plot_comparative_roc, plot_comparative_bar_chart
    from benchmark import evaluate_all_models
except ImportError:
    print("Please restart the runtime/kernel after installing dependencies if you get an ImportError, then run this cell again.")
""")

data_cell_md = nbf.v4.new_markdown_cell("""## 1. Data Acquisition and Preprocessing
The fundamental requirement for robust feature extraction is high-quality, standardized data. This section programmatically fetches the required datasets and initializes the data ingestion pipelines. The preprocessing pipeline applies requisite transformations including uniform image resizing (224x224), min-max normalization, and spatial augmentations to mitigate overfitting and enhance model generalization across pathological variations.""")
data_cell_code = nbf.v4.new_code_cell("""# Download Datasets via Kaggle API
!python src/download_data.py

# Load Malaria Data
print("\\nLoading Malaria Dataset...")
base_dir = os.path.abspath('.')
malaria_train, malaria_val = load_malaria_data(base_dir, batch_size=32)

# Load TB Data (Uncomment to use TB data instead)
# print("Loading TB Dataset...")
# tb_train, tb_val = load_tb_data(base_dir, batch_size=32)
""")

model_cell_md = nbf.v4.new_markdown_cell("""## 2. Model Architecture and Attention Mechanism Initialization
To empirically evaluate the efficacy of spatial and channel attention mechanisms, this framework implements five distinct architectures. Transfer learning (via ImageNet pre-trained weights) is utilized for standard architectures, and a custom Convolutional Block Attention Module (CBAM) is integrated into each. By default, the lightweight MobileNetV2 architecture is initialized for rapid experimental validation.""")
model_cell_code = nbf.v4.new_code_cell("""# Select Model Architecture for the Current Experimental Run
# model = build_custom_cnn_attention()
# model = build_resnet50_attention()
# model = build_vgg16_attention()
model = build_mobilenetv2_attention()
# model = build_densenet121_attention()

# Compile
model = compile_model(model, learning_rate=1e-4)
model.summary()
""")

train_cell_md = nbf.v4.new_markdown_cell("""## 3. Network Training Phase
During the initial training phase, the base convolutional layers are frozen to preserve the pre-trained ImageNet feature hierarchies. The network optimizes only the newly appended classification head and the integrated CBAM attention layers. The optimization leverages the Adam algorithm with a categorical cross-entropy objective function, monitored dynamically via Keras callbacks (EarlyStopping and ModelCheckpoint) to prevent overfitting and guarantee convergence to the global minima.""")
train_cell_code = nbf.v4.new_code_cell("""# Define the target dataset for dynamic checkpointing
dataset_name = "malaria" # CHANGE THIS TO "tb" IF TRAINING ON TUBERCULOSIS

# Generate a dynamic save path so models don't overwrite each other
save_path = f"best_{dataset_name}_{model.name}.h5"
print(f"Model will be saved to: {save_path}")

# Train the model
# Ensure you change train_data and val_data to tb_train and tb_val if training for Tuberculosis
history = train_model(
    model, 
    train_data=malaria_train, 
    val_data=malaria_val, 
    epochs=15, 
    model_path=save_path
)

plot_training_history(history, model_name=f"{model.name} ({dataset_name})")
""")

finetune_cell_md = nbf.v4.new_markdown_cell("""## 4. Fine-Tuning and Optimization
To adapt the generalized ImageNet features specifically to medical radiographic and microscopic textures, the top spatial layers of the base network are unfreezed. Training is resumed using a strictly reduced learning rate (e.g., 1e-5) to avoid destructive catastrophic forgetting while allowing the high-level feature extractors to align precisely with the pathological dataset.""")
finetune_cell_code = nbf.v4.new_code_cell("""# Execute Fine-Tuning on the top 20 architectural layers
history_ft = unfreeze_and_finetune(
    model, 
    train_data=malaria_train, 
    val_data=malaria_val, 
    layers_to_unfreeze=20, 
    epochs=10, 
    learning_rate=1e-5
)

plot_training_history(history_ft, model_name=f"{model.name}_FineTuned")
""")

gradcam_cell_md = nbf.v4.new_markdown_cell("""## 5. Interpretability Analysis via Grad-CAM
In medical image analysis, model transparency is paramount. The Gradient-weighted Class Activation Mapping (Grad-CAM) algorithm is employed to visually validate the model's spatial decision-making process. By projecting the gradients of the target concept onto the final convolutional feature maps, we can confirm whether the network and its attention modules are appropriately fixated on pathological anomalies (e.g., plasmodium parasites or pulmonary opacities) rather than background artifacts.""")
gradcam_cell_code = nbf.v4.new_code_cell("""# Execute Grad-CAM Interpretability Analysis (Requires a specific target image path)
import numpy as np
from utils import get_img_array, make_gradcam_heatmap, display_gradcam

# img_path = 'data/malaria/cell_images/cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_162.png'
# img_array = get_img_array(img_path, size=(224, 224))

# For MobileNetV2, the last conv layer is often 'out_relu' or you can inspect model.summary()
# last_conv_layer_name = 'out_relu'

# heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
# display_gradcam(img_path, heatmap)
""")

nb.cells = [
    title_cell, setup_cell, 
    data_cell_md, data_cell_code,
    model_cell_md, model_cell_code,
    train_cell_md, train_cell_code,
    finetune_cell_md, finetune_cell_code,
    gradcam_cell_md, gradcam_cell_code
]

eval_cell_md = nbf.v4.new_markdown_cell("""## 6. Comprehensive Statistical Evaluation
Rigorous statistical evaluation ensures the model's reliability in clinical scenarios. This section restores the most optimal weights observed during the training and fine-tuning epochs. It then subjects the model to the validation cohort to extract a comprehensive suite of metrics including Accuracy, Sensitivity (Recall), Specificity, Precision, F1-Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE). 

Furthermore, provisions for McNemar's statistical test are included to assess whether the discordance between the predictions of two distinct architectures is statistically significant.""")
eval_cell_code = nbf.v4.new_code_cell("""# Load the optimal weights preserved by the ModelCheckpoint callback
# model.load_weights(save_path)

# Get true labels and predictions for validation set
# y_true = malaria_val.classes
# y_pred_probs_m1 = model.predict(malaria_val)

# evaluate_comprehensive_metrics(y_true, y_pred_probs_m1)

# ---
# If you are returning to this notebook later and want to load a completely saved model from disk:
# import tensorflow as tf
# from attention import cbam_block, channel_attention, spatial_attention
# custom_objects = {'cbam_block': cbam_block, 'channel_attention': channel_attention, 'spatial_attention': spatial_attention}
# loaded_model = tf.keras.models.load_model(save_path, custom_objects=custom_objects)
# ---

# If you have a second model (e.g. VGG16) to compare:
# y_pred_probs_m2 = model_vgg.predict(malaria_val)
# perform_mcnemar_test(y_true, y_pred_probs_m1, y_pred_probs_m2)
""")

nb.cells.extend([eval_cell_md, eval_cell_code])

benchmark_cell_md = nbf.v4.new_markdown_cell("""## 7. Comparative Benchmarking (Results Aggregation)
The concluding phase addresses the requirements of the thesis's Results and Discussion chapters. This automated benchmarking suite loads all experimentally trained models (assuming multiple architectures have been evaluated sequentially) into memory. It conducts a unified evaluation across the test set, measuring classification metrics alongside critical deployment constraints such as computational parameter count, file size, and per-image inference latency. The outputs (comparative CSV tables, overlaid ROC curves, and bar charts) are explicitly designed for direct academic publication.""")
benchmark_cell_code = nbf.v4.new_code_cell("""# Initialize a dictionary mapping architecture names to their local persistent checkpoints
models_dict = {
    "Custom CNN": f"best_{dataset_name}_Custom_CNN_Attention.h5",
    "ResNet50": f"best_{dataset_name}_ResNet50_Attention.h5",
    "VGG16": f"best_{dataset_name}_VGG16_Attention.h5",
    "MobileNetV2": f"best_{dataset_name}_MobileNetV2_Attention.h5",
    "DenseNet121": f"best_{dataset_name}_DenseNet121_Attention.h5"
}

# 1. Generate DataFrame with Latency, Params, Size, and Classification Metrics
# df, y_true, predictions_dict = evaluate_all_models(models_dict, malaria_val, dataset_name, output_csv=f"comparative_results_{dataset_name}.csv")

# 2. Display the generated table (Copy this into your Results Chapter)
# display(df)

# 3. Plot Overlaid ROC Curves
# plot_comparative_roc(y_true, predictions_dict, title=f"Comparative ROC Curves ({dataset_name.upper()})")

# 4. Plot Comparative Bar Charts
# plot_comparative_bar_chart(df, metric='Accuracy')
# plot_comparative_bar_chart(df, metric='F1-Score')
# plot_comparative_bar_chart(df, metric='Latency (ms/image)')
""")

nb.cells.extend([benchmark_cell_md, benchmark_cell_code])

with open('c:/MyProjects/ml/main.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Jupyter Notebook created successfully!")
