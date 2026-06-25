# Attention-Improved Deep Learning Framework for Malaria and Tuberculosis Detection

This repository contains the implementation of an attention-improved deep learning framework designed to automate the diagnosis of Malaria and Tuberculosis. By integrating **Convolutional Block Attention Modules (CBAM)** into pre-trained architectures (including Custom CNN, ResNet50, VGG16, MobileNetV2, and DenseNet121), this project enhances diagnostic accuracy and interpretability while evaluating computational efficiency for deployment in resource-constrained healthcare settings like Nigeria.

## Project Structure
- `src/download_data.py`: Uses the Kaggle API to programmatically download and unzip the required NIH Malaria and Tuberculosis datasets.
- `src/data_loader.py`: Sets up TensorFlow `ImageDataGenerator` for image resizing, normalization, histogram equalization, and augmentation.
- `src/attention.py`: Contains the custom implementation of the Spatial and Channel attention mechanisms (CBAM).
- `src/models.py`: Definitions for the 5 comparative architectures utilizing ImageNet pre-trained weights and the custom attention layer.
- `src/train.py`: Modular training loops containing Early Stopping, Checkpointing, and Learning Rate scheduling logic.
- `src/utils.py`: Utilities for plotting training history and generating **Grad-CAM heatmaps** for clinical explainability.
- `main.ipynb`: The primary Jupyter Notebook designed to orchestrate the entire pipeline (data loading, model compilation, training, and visualization).

## How to Run (Colab / Azure ML)

1. **Clone the repository** to your cloud instance.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Data Acquisition**: 
   Ensure you have your `kaggle.json` API token uploaded to your environment (`~/.kaggle/kaggle.json`), then run:
   ```bash
   python src/download_data.py
   ```
4. **Execution**:
   Open `main.ipynb`, select your desired architecture, and run the cells sequentially to train and evaluate the models.

## Original Research
This framework is based on the research project: *Attention-Improved Deep Learning Framework for Malaria and Tuberculosis Detection and Classification in Nigeria* by Muhammed Toheeb Abdulraheem.
