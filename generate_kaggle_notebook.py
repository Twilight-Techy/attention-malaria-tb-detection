import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python"
    }
}

title_cell = nbf.v4.new_markdown_cell("""# Automated End-to-End Deep Learning Framework for Malaria and Tuberculosis (KAGGLE EDITION)
**Author:** Muhammed Toheeb Abdulraheem

**Abstract:**
This notebook is specifically tailored for the **Kaggle Notebook** environment. It utilizes Kaggle's native dataset attachment feature for instantaneous data loading (zero download time) and leverages the `/kaggle/working/` directory for persistent artifact storage.
""")

kaggle_setup_cell = nbf.v4.new_code_cell("""# =========================================================================
# 1. KAGGLE ENVIRONMENT SETUP & GITHUB CLONE
# =========================================================================
# Kaggle mounts your datasets securely as read-only at /kaggle/input/
# We will clone your repository into the writable /kaggle/working/ directory
# and set up a persistent Thesis_Results folder for your model weights!

import os

print("Setting up Kaggle Workspace...")

# Kaggle's writable directory
working_dir = '/kaggle/working'
repo_name = 'attention-malaria-tb-detection'
repo_path = os.path.join(working_dir, repo_name)
results_dir = os.path.join(working_dir, 'Thesis_Results')

# Move into working directory
get_ipython().run_line_magic('cd', working_dir)

# Clone repository if it doesn't exist, otherwise pull latest changes
if not os.path.exists(repo_name):
    get_ipython().system('git clone https://github.com/Twilight-Techy/attention-malaria-tb-detection.git')
else:
    get_ipython().system('cd attention-malaria-tb-detection && git pull')

# Move into the repository
get_ipython().run_line_magic('cd', repo_name)

# Create persistent results folder
os.makedirs(results_dir, exist_ok=True)

# Auto-recover weights and logs from any mounted previous Kaggle versions
print("Restoring previous training weights and logs, preserving exact folder structure...")
import glob
dataset_paths = glob.glob('/kaggle/input/**/Thesis_Results', recursive=True)
if dataset_paths:
    source_dir = dataset_paths[0]
    print(f"Found previous results at {source_dir}, copying...")
    get_ipython().system(f'cp -rn {source_dir}/* {results_dir}/ 2>/dev/null || true')
else:
    print("Could not find previous Thesis_Results dataset.")

print("\\nRecovered files in Thesis_Results:")
get_ipython().system(f'ls -lh {results_dir}')

# Copy previously trained weights from Thesis_Results into the repo folder for Keras to resume
get_ipython().system(f'cp -r {results_dir}/* . 2>/dev/null || true')

# Set up a background process to continuously sync your progress into Thesis_Results
get_ipython().system_raw(f'while true; do cp *.h5 *.done *.phase1_done *.csv *.png {results_dir}/ 2>/dev/null; cp -r backup_* {results_dir}/ 2>/dev/null; sleep 120; done &')

print("Kaggle Workspace initialized! Background sync to /kaggle/working/Thesis_Results is active.")
""")

setup_cell = nbf.v4.new_code_cell("""import sys
import os
import gc
import subprocess
import tensorflow.keras.backend as K
import pandas as pd
import importlib
from IPython.display import display

# 1. Install Dependencies
print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
importlib.invalidate_caches()

# 2. Setup Python Path
# Use insert(0) to prevent Kaggle's pre-installed packages (like 'benchmark') from shadowing our local files!
sys.path.insert(0, os.path.abspath('src'))

from data_loader import load_malaria_data, load_tb_data, load_full_production_dataset
from models import build_custom_cnn_attention, build_resnet50_attention, build_vgg16_attention, build_mobilenetv2_attention, build_densenet121_attention
from train import compile_model, train_model, unfreeze_and_finetune
from utils import plot_training_history, plot_comparative_roc, plot_comparative_bar_chart
from benchmark import evaluate_all_models
    
base_dir = os.path.abspath('.')
""")

loop_cell_md = nbf.v4.new_markdown_cell("""## Master Execution Pipeline
This loop sequentially tackles Malaria, then Tuberculosis. For each disease, it trains all 5 architectures, saves the optimal `.h5` model files, and outputs the final comparative benchmark metrics.
""")

loop_cell_code = nbf.v4.new_code_cell("""datasets_to_run = ["malaria", "tb"]
architectures_to_run = [
    ("MobileNetV2", build_mobilenetv2_attention),
    ("Custom CNN", build_custom_cnn_attention),
    ("VGG16", build_vgg16_attention),
    ("ResNet50", build_resnet50_attention),
    ("DenseNet121", build_densenet121_attention)
]

for dataset_name in datasets_to_run:
    print(f"\\n{'='*60}\\nSTARTING EXPERIMENTAL PIPELINE FOR: {dataset_name.upper()}\\n{'='*60}")
    
    # 1. Load Data (Loads directly from Kaggle Input to NVMe Cache!)
    if dataset_name == "malaria":
        # Adjust this path if the dataset is mounted differently in Kaggle
        kaggle_malaria_path = '/kaggle/input/datasets/iarunava/cell-images-for-detecting-malaria/cell_images/cell_images'
        if not os.path.exists(kaggle_malaria_path):
            kaggle_malaria_path = '/kaggle/input/datasets/iarunava/cell-images-for-detecting-malaria/cell_images'
            if not os.path.exists(kaggle_malaria_path):
                kaggle_malaria_path = '/kaggle/input/datasets/iarunava/cell-images-for-detecting-malaria'
        train_data, val_data, test_data = load_malaria_data(base_dir, data_dir=kaggle_malaria_path, batch_size=32)
    else:
        # Adjust this path if the dataset is mounted differently in Kaggle
        kaggle_tb_path = '/kaggle/input/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database'
        if not os.path.exists(kaggle_tb_path):
            kaggle_tb_path = '/kaggle/input/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset'
        train_data, val_data, test_data = load_tb_data(base_dir, data_dir=kaggle_tb_path, batch_size=32)
        
    # 1.5. Exploratory Data Analysis
    print(f"\\n--- [ Exploratory Data Analysis for {dataset_name.upper()} ] ---")
    from utils import plot_class_distribution, plot_sample_images
    plot_class_distribution(train_data, dataset_name, save_path=f"eda_distribution_{dataset_name}.png")
    plot_sample_images(train_data, dataset_name, save_path=f"eda_samples_{dataset_name}.png")
        
    models_dict = {}
    
    # 2. Iterate and Train Models
    for model_name, model_builder in architectures_to_run:
        print(f"\\n--- [ Training {model_name} on {dataset_name.upper()} ] ---")
        
        # CRITICAL: Clear GPU VRAM before instantiating the next model
        K.clear_session()
        gc.collect()
        
        # Build and Compile
        model = model_builder()
        model = compile_model(model, learning_rate=1e-4)
        save_path = f"best_{dataset_name}_{model.name}.h5"
        models_dict[model_name] = save_path
        
        # FAULT TOLERANCE: Skip if completely finished previously
        completion_marker = f"{save_path}.done"
        phase1_marker = f"{save_path}.phase1_done"
        
        if os.path.exists(completion_marker):
            print(f"\\n[RESUME] Model {model.name} fully completed previously. Skipping to next!")
            continue
        elif os.path.exists(save_path) and os.path.exists(phase1_marker):
            print(f"\\n[RESUME] Found successful Phase 1 weights ({save_path}) and phase 1 completion marker.")
            print(f"Loading weights and jumping straight to Phase 2 (Fine-Tuning) to save time!")
            model.load_weights(save_path)
            phase1_completed = True
        elif os.path.exists(save_path) and not os.path.exists(phase1_marker):
            print(f"\\n[RESUME] Found partial Phase 1 weights ({save_path}) but NO phase 1 completion marker.")
            print(f"Phase 1 crashed mid-training. Re-invoking Phase 1 (Keras will automatically resume from the exact epoch using your Backup Folder).")
            phase1_completed = False
        else:
            phase1_completed = False
            
        log_path = f"training_log_{dataset_name}_{model.name}.csv"
        
        if not phase1_completed:
            # Train (Base Layers Frozen)
            print(f"\\nPhase 1: Freezing Base Layers and Training Classification Head")
            train_model(model, train_data, val_data, epochs=15, model_path=save_path, csv_log_path=log_path)
            
            # Mark Phase 1 as completely finished
            with open(phase1_marker, 'w') as f:
                f.write("phase 1 complete")
        
        # Fine-Tune (Unfreezing Top Layers)
        print(f"\\nPhase 2: Fine-Tuning Top Feature Extractors")
        unfreeze_and_finetune(model, train_data, val_data, layers_to_unfreeze=20, epochs=10, learning_rate=1e-5, csv_log_path=log_path, model_path=save_path, initial_epoch=15)
        
        # Mark as completely finished
        with open(completion_marker, 'w') as f:
            f.write("training and finetuning complete")
            
    # 3. Benchmark Dataset (Generates Results Chapter Tables/Graphs)
    print(f"\\n--- [ Benchmarking All {dataset_name.upper()} Architectures ] ---")
    df, y_true, predictions_dict = evaluate_all_models(models_dict, test_data, dataset_name, output_csv=f"comparative_results_{dataset_name}.csv")
    
    print("\\nFinal Comparative DataFrame:")
    display(df)
    
    print("\\nGenerating ROC Curves...")
    plot_comparative_roc(y_true, predictions_dict, title=f"Comparative ROC Curves ({dataset_name.upper()})", save_path=f"comparative_roc_{dataset_name}.png")
    
    print("\\nGenerating F1-Score Bar Chart...")
    plot_comparative_bar_chart(df, metric='F1-Score', title=f"F1-Score Comparison ({dataset_name.upper()})", save_path=f"comparative_f1_{dataset_name}.png")
    
    # 4. Clean up Dataset from RAM before loading the next disease
    del train_data
    del val_data
    del test_data
    gc.collect()
    
print("\\n\\nPIPELINE COMPLETE. All experiments successfully finished and data saved.")
""")

production_md = nbf.v4.new_markdown_cell("""## 8. Final Production Deployment Pipeline
**[POST-THESIS ONLY]** Once you have completed your academic benchmarks above, you will identify your absolute best-performing architecture for Malaria and Tuberculosis. 

To prepare for actual hospital deployment (Section 1.5.v of the report), you should retrain that winning architecture on **100% of the available data** (Train + Val + Test merged into one).
Below is the template to generate your final deployment `.h5` model. Uncomment and run it when ready!
""")

production_code = nbf.v4.new_code_cell("""'''
# Example: Assuming MobileNetV2 won the benchmark for Malaria
deployment_dataset = "malaria"
print(f"\\n{'='*50}\\nSTARTING PRODUCTION DEPLOYMENT PIPELINE FOR: {deployment_dataset.upper()}\\n{'='*50}")

# 1. Load 100% of data (NO SPLITS)
# Adjust this path if the dataset is mounted differently in Kaggle
kaggle_malaria_path = '/kaggle/input/datasets/iarunava/cell-images-for-detecting-malaria/cell_images/cell_images'
if not os.path.exists(kaggle_malaria_path):
    kaggle_malaria_path = '/kaggle/input/datasets/iarunava/cell-images-for-detecting-malaria/cell_images'
    if not os.path.exists(kaggle_malaria_path):
        kaggle_malaria_path = '/kaggle/input/datasets/iarunava/cell-images-for-detecting-malaria'
    
production_data = load_full_production_dataset(base_dir, data_dir=kaggle_malaria_path, dataset_name=deployment_dataset, batch_size=32)

# 2. Build winning architecture
K.clear_session()
final_model = build_mobilenetv2_attention()
final_model = compile_model(final_model, learning_rate=1e-4)
final_save_path = f"FINAL_PRODUCTION_{deployment_dataset}_MobileNetV2.h5"

# 3. Train on 100% data (Fixed Epochs, No Early Stopping since there is no val_data)
# Note: For production, we manually enforce the epoch count since Early Stopping requires a validation set.
print(f"\\nPhase 1: Freezing Base Layers...")
final_model.fit(production_data, epochs=15) # Fit directly without val_data

print(f"\\nPhase 2: Fine-Tuning Top Layers...")
final_model.trainable = True
# Unfreeze top 20 layers
for layer in final_model.layers[:-20]:
    layer.trainable = False
final_model = compile_model(final_model, learning_rate=1e-5)
final_model.fit(production_data, epochs=10)

# 4. Save Final Production Model
final_model.save(final_save_path)
print(f"\\nSUCCESS! Final deployment model saved to: {final_save_path}")
'''
""")

nb.cells = [title_cell, kaggle_setup_cell, setup_cell, loop_cell_md, loop_cell_code, production_md, production_code]

with open('kaggle_main.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Kaggle Jupyter Notebook Master Pipeline created successfully!")
