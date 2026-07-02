import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np
import splitfolders

def advanced_preprocessing(img):
    """
    Applies Histogram Equalization (CLAHE) and Noise Reduction (Gaussian Blur) 
    as specified in the project report.
    """
    # Ensure image is in uint8 format for OpenCV processing
    if img.dtype != np.uint8:
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_uint8 = img

    # Convert to LAB color space to apply CLAHE to the L channel
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    
    # Merge and convert back to RGB
    merged = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    
    # Apply Noise Reduction (Gaussian Blur)
    denoised_img = cv2.GaussianBlur(enhanced_img, (3, 3), 0)
    
    return denoised_img.astype(np.float32)

def create_data_generators(data_dir, target_size=(224, 224), batch_size=32):
    """
    Physically splits the data into Train (70%), Validation (15%), and Test (15%) subsets.
    Then creates distinct generators for each split to prevent data leakage.
    """
    output_dir = data_dir + "_split"
    
    if not os.path.exists(output_dir):
        print(f"Splitting dataset into Train/Val/Test subsets in {output_dir}...")
        splitfolders.ratio(data_dir, output=output_dir, seed=42, ratio=(0.7, 0.15, 0.15), group_prefix=None)
    
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    # Data Augmentation specific to the report's requirements
    train_datagen = ImageDataGenerator(
        rescale=1./255, 
        preprocessing_function=advanced_preprocessing,
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=[0.9, 1.1],
        brightness_range=[0.8, 1.2],
        width_shift_range=0.05, 
        height_shift_range=0.05
    )

    # For validation and test, strictly only rescaling and preprocessing (No Augmentation)
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=advanced_preprocessing
    )

    print(f"Loading split generators from {output_dir}...")
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    val_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def load_malaria_data(base_dir, batch_size=32):
    data_dir = os.path.join(base_dir, "data", "malaria", "cell_images", "cell_images")
    return create_data_generators(data_dir, batch_size=batch_size)

def load_tb_data(base_dir, batch_size=32):
    # Note: the TB dataset structure from Kaggle might vary. 
    # Update the inner path depending on how it unzips.
    data_dir = os.path.join(base_dir, "data", "tuberculosis", "TB_Chest_Radiography_Database")
    return create_data_generators(data_dir, batch_size=batch_size)
