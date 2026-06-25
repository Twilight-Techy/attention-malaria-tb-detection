import nbformat as nbf

nb = nbf.v4.new_notebook()

title_cell = nbf.v4.new_markdown_cell("""# Attention-Improved Deep Learning Framework for Malaria and Tuberculosis
**Author:** Muhammed Toheeb Abdulraheem

This notebook orchestrates the training, evaluation, and interpretability of 5 deep learning architectures using custom CBAM attention layers.
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
    from utils import plot_training_history, make_gradcam_heatmap, display_gradcam, evaluate_comprehensive_metrics, perform_mcnemar_test
except ImportError:
    print("Please restart the runtime/kernel after installing dependencies if you get an ImportError, then run this cell again.")
""")

data_cell_md = nbf.v4.new_markdown_cell("## 1. Download and Load Datasets\nThis will automatically download the datasets if they are not already present.")
data_cell_code = nbf.v4.new_code_cell("""# Download Data
!python src/download_data.py

# Load Malaria Data
print("\\nLoading Malaria Dataset...")
base_dir = os.path.abspath('.')
malaria_train, malaria_val = load_malaria_data(base_dir, batch_size=32)

# Load TB Data (Uncomment to use TB data instead)
# print("Loading TB Dataset...")
# tb_train, tb_val = load_tb_data(base_dir, batch_size=32)
""")

model_cell_md = nbf.v4.new_markdown_cell("## 2. Build Models with Attention\nWe can instantiate any of the 5 models here. Let's start with the lightweight MobileNetV2 for fast cloud execution.")
model_cell_code = nbf.v4.new_code_cell("""# Choose your architecture
# model = build_custom_cnn_attention()
# model = build_resnet50_attention()
# model = build_vgg16_attention()
model = build_mobilenetv2_attention()
# model = build_densenet121_attention()

# Compile
model = compile_model(model, learning_rate=1e-4)
model.summary()
""")

train_cell_md = nbf.v4.new_markdown_cell("## 3. Train Model\nTraining the top layers while the base convolutional layers are frozen.")
train_cell_code = nbf.v4.new_code_cell("""# Train the model
# Change malaria_train to tb_train if training for Tuberculosis
history = train_model(
    model, 
    train_data=malaria_train, 
    val_data=malaria_val, 
    epochs=15, 
    model_path='best_malaria_mobilenet_attention.h5'
)

plot_training_history(history, model_name=model.name)
""")

finetune_cell_md = nbf.v4.new_markdown_cell("## 4. Fine-Tuning\nUnfreeze the top layers and fine-tune with a lower learning rate.")
finetune_cell_code = nbf.v4.new_code_cell("""# Fine-tune the top 20 layers
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

gradcam_cell_md = nbf.v4.new_markdown_cell("## 5. Interpretability: Grad-CAM\nVisualizing what the attention layers are focusing on.")
gradcam_cell_code = nbf.v4.new_code_cell("""# Example Grad-CAM execution (Requires an image path)
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

eval_cell_md = nbf.v4.new_markdown_cell("## 6. Comprehensive Evaluation & McNemar's Test\nCalculate Specificity, F1, MAE, RMSE, and statistically compare two models.")
eval_cell_code = nbf.v4.new_code_cell("""# Get true labels and predictions for validation set
# y_true = malaria_val.classes
# y_pred_probs_m1 = model.predict(malaria_val)

# evaluate_comprehensive_metrics(y_true, y_pred_probs_m1)

# If you have a second model (e.g. VGG16) to compare:
# y_pred_probs_m2 = model_vgg.predict(malaria_val)
# perform_mcnemar_test(y_true, y_pred_probs_m1, y_pred_probs_m2)
""")

nb.cells.extend([eval_cell_md, eval_cell_code])

with open('c:/MyProjects/ml/main.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Jupyter Notebook created successfully!")
