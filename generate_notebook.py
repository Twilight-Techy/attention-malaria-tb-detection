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

title_cell = nbf.v4.new_markdown_cell("""# Automated End-to-End Deep Learning Framework for Malaria and Tuberculosis
**Author:** Muhammed Toheeb Abdulraheem

**Abstract:**
This notebook contains an **Automated Master Execution Loop**. It is designed to autonomously download datasets, load them into memory sequentially, and train all five attention-improved architectures (Custom CNN, ResNet50, VGG16, MobileNetV2, DenseNet121) across both the Malaria and Tuberculosis datasets. To prevent GPU Out-of-Memory (OOM) errors, it implements strict garbage collection and Keras session clearing between each architecture. 

    It concludes by generating the final comparative CSV tables, ROC curves, and bar charts required for the Results chapter.
""")

colab_setup_cell = nbf.v4.new_code_cell("""# =========================================================================
# 1. OPTIONAL: COLAB ENVIRONMENT SETUP & DRIVE SYNC
# =========================================================================
# If running in Google Colab, this cell will safely mount your Google Drive, 
# pull any previously trained weights to resume from crashes, and start a 
# background loop that saves your progress every 2 minutes!

try:
    from google.colab import drive
    import os
    print("Google Colab environment detected. Mounting Drive...")
    drive.mount('/content/drive')
    
    # Auto-resume pulls
    get_ipython().system('mkdir -p /content/drive/MyDrive/Thesis_Results')
    get_ipython().system('cp /content/drive/MyDrive/Thesis_Results/*.h5 . 2>/dev/null || true')
    get_ipython().system('cp /content/drive/MyDrive/Thesis_Results/*.done . 2>/dev/null || true')
    get_ipython().system('cp /content/drive/MyDrive/Thesis_Results/*.phase1_done . 2>/dev/null || true')
    get_ipython().system('cp -r /content/drive/MyDrive/Thesis_Results/backup_* . 2>/dev/null || true')
    
    # Background sync
    get_ipython().system_raw('while true; do cp *.h5 *.done *.phase1_done *.csv *.png /content/drive/MyDrive/Thesis_Results/ 2>/dev/null; cp -r backup_* /content/drive/MyDrive/Thesis_Results/ 2>/dev/null; sleep 120; done &')
    print("Background Drive Sync initialized! Your progress is safe.")
    
except ImportError:
    print("Local environment detected. Skipping Google Drive mount and sync.")
""")

setup_cell = nbf.v4.new_code_cell("""import sys
import os
import gc
import subprocess
import tensorflow.keras.backend as K
import getpass
import pandas as pd
from IPython.display import display

# 1. Install Dependencies
print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])

# 2. Setup Kaggle Credentials
print("\\nAuthenticating with Kaggle...")
if 'KAGGLE_USERNAME' not in os.environ:
    os.environ['KAGGLE_USERNAME'] = input("Please enter your Kaggle Username (e.g. johndoe): ")
os.environ['KAGGLE_KEY'] = 'c73c266a0b891d30683588637504fc56' # Hardcoded API Key provided by user

# 3. Setup Python Path
sys.path.append(os.path.abspath('src'))

try:
    from data_loader import load_malaria_data, load_tb_data, load_full_production_dataset
    from models import build_custom_cnn_attention, build_resnet50_attention, build_vgg16_attention, build_mobilenetv2_attention, build_densenet121_attention
    from train import compile_model, train_model, unfreeze_and_finetune
    from utils import plot_training_history, plot_comparative_roc, plot_comparative_bar_chart
    from benchmark import evaluate_all_models
except ImportError:
    print("Dependencies installed. Please restart the Jupyter kernel and run this cell again.")
""")

data_cell = nbf.v4.new_code_cell("""# Download both datasets to the local VM disk
import subprocess
print("Running data downloader...")
subprocess.check_call([sys.executable, "src/download_data.py"])
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
            print(f"Phase 1 crashed mid-training. Restarting Phase 1 from scratch to prevent corrupt weights.")
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
        unfreeze_and_finetune(model, train_data, val_data, layers_to_unfreeze=20, epochs=10, learning_rate=1e-5, csv_log_path=log_path)
        
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
production_data = load_full_production_dataset(base_dir, dataset_name=deployment_dataset, batch_size=32)

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

nb.cells = [title_cell, colab_setup_cell, setup_cell, data_cell, loop_cell_md, loop_cell_code, production_md, production_code]

with open('c:/MyProjects/ml/main.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Jupyter Notebook Master Pipeline created successfully!")
