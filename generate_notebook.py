import nbformat as nbf

nb = nbf.v4.new_notebook()

title_cell = nbf.v4.new_markdown_cell("""# Automated End-to-End Deep Learning Framework for Malaria and Tuberculosis
**Author:** Muhammed Toheeb Abdulraheem

**Abstract:**
This notebook contains an **Automated Master Execution Loop**. It is designed to autonomously download datasets, load them into memory sequentially, and train all five attention-improved architectures (Custom CNN, ResNet50, VGG16, MobileNetV2, DenseNet121) across both the Malaria and Tuberculosis datasets. To prevent GPU Out-of-Memory (OOM) errors, it implements strict garbage collection and Keras session clearing between each architecture. 

It concludes by generating the final comparative CSV tables, ROC curves, and bar charts required for the Results chapter.
""")

setup_cell = nbf.v4.new_code_cell("""import sys
import os
import gc
import tensorflow.keras.backend as K
import getpass
import pandas as pd
from IPython.display import display

# 1. Install Dependencies
print("Installing dependencies...")
!pip install -q -r requirements.txt

# 2. Setup Kaggle Credentials
print("\\nAuthenticating with Kaggle...")
if 'KAGGLE_USERNAME' not in os.environ:
    os.environ['KAGGLE_USERNAME'] = input("Please enter your Kaggle Username (e.g. johndoe): ")
os.environ['KAGGLE_KEY'] = 'c73c266a0b891d30683588637504fc56' # Hardcoded API Key provided by user

# 3. Setup Python Path
sys.path.append(os.path.abspath('src'))

try:
    from data_loader import load_malaria_data, load_tb_data
    from models import build_custom_cnn_attention, build_resnet50_attention, build_vgg16_attention, build_mobilenetv2_attention, build_densenet121_attention
    from train import compile_model, train_model, unfreeze_and_finetune
    from utils import plot_training_history, plot_comparative_roc, plot_comparative_bar_chart
    from benchmark import evaluate_all_models
except ImportError:
    print("Dependencies installed. Please restart the Jupyter kernel and run this cell again.")
""")

data_cell = nbf.v4.new_code_cell("""# Download both datasets to the local VM disk
!python src/download_data.py
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
    
    # 1. Load Data (Loads directly into RAM)
    if dataset_name == "malaria":
        train_data, val_data, test_data = load_malaria_data(base_dir, batch_size=32)
    else:
        train_data, val_data, test_data = load_tb_data(base_dir, batch_size=32)
        
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
        if os.path.exists(completion_marker):
            print(f"\\n[RESUME] Model {model.name} fully completed previously. Skipping to next!")
            continue
        elif os.path.exists(save_path):
            print(f"\\n[RESUME] Found partial {save_path} but no completion marker. Restarting training for this model to ensure full completion.")
        
        # Train (Base Layers Frozen)
        print(f"\\nPhase 1: Freezing Base Layers and Training Classification Head")
        train_model(model, train_data, val_data, epochs=15, model_path=save_path)
        
        # Fine-Tune (Unfreezing Top Layers)
        print(f"\\nPhase 2: Fine-Tuning Top Feature Extractors")
        unfreeze_and_finetune(model, train_data, val_data, layers_to_unfreeze=20, epochs=10, learning_rate=1e-5)
        
        # Mark as completely finished
        with open(completion_marker, 'w') as f:
            f.write("training and finetuning complete")
            
    # 3. Benchmark Dataset (Generates Results Chapter Tables/Graphs)
    print(f"\\n--- [ Benchmarking All {dataset_name.upper()} Architectures ] ---")
    df, y_true, predictions_dict = evaluate_all_models(models_dict, test_data, dataset_name, output_csv=f"comparative_results_{dataset_name}.csv")
    
    print("\\nFinal Comparative DataFrame:")
    display(df)
    
    print("\\nGenerating ROC Curves...")
    plot_comparative_roc(y_true, predictions_dict, title=f"Comparative ROC Curves ({dataset_name.upper()})")
    
    print("\\nGenerating F1-Score Bar Chart...")
    plot_comparative_bar_chart(df, metric='F1-Score', title=f"F1-Score Comparison ({dataset_name.upper()})")
    
    # 4. Clean up Dataset from RAM before loading the next disease
    del train_data
    del val_data
    del test_data
    gc.collect()
    
print("\\n\\nPIPELINE COMPLETE. All experiments successfully finished and data saved.")
""")

nb.cells = [title_cell, setup_cell, data_cell, loop_cell_md, loop_cell_code]

with open('c:/MyProjects/ml/main.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Jupyter Notebook Master Pipeline created successfully!")
