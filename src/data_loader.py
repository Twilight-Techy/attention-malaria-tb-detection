import tensorflow as tf
import os
import cv2
import numpy as np
import splitfolders
from tensorflow.keras import layers

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

def process_batch(images, labels):
    def apply_opencv_batch(imgs):
        processed = []
        for img in imgs:
            processed.append(advanced_preprocessing(img))
        return np.array(processed, dtype=np.float32)
        
    processed_images = tf.numpy_function(apply_opencv_batch, [images], tf.float32)
    processed_images.set_shape((None, 224, 224, 3))
    
    # Normalize to [0, 1]
    processed_images = processed_images / 255.0
    
    # Flatten label shape from (batch, 1) to (batch,) to perfectly mimic ImageDataGenerator
    labels = tf.reshape(labels, [-1])
    
    return processed_images, labels

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(15/360),
    layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
    layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
    layers.RandomBrightness(factor=0.2),
])

def train_augment(images, labels):
    images = data_augmentation(images, training=True)
    return images, labels

def create_data_generators(data_dir, target_size=(224, 224), batch_size=32):
    """
    Physically splits the data into Train (70%), Validation (15%), and Test (15%) subsets.
    Then creates highly optimized tf.data pipelines for each split.
    """
    output_dir = data_dir + "_split"
    
    if not os.path.exists(output_dir):
        print(f"Splitting dataset into Train/Val/Test subsets in {output_dir}...")
        splitfolders.ratio(data_dir, output=output_dir, seed=42, ratio=(0.7, 0.15, 0.15), group_prefix=None)
    
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    print(f"Loading tf.data pipelines from {output_dir}...")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=target_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=True
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=target_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=target_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False
    )
    
    # Parallel CPU processing
    train_ds = train_ds.map(process_batch, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(train_augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    val_ds = val_ds.map(process_batch, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    test_ds = test_ds.map(process_batch, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds

def load_malaria_data(base_dir, batch_size=32):
    data_dir = os.path.join(base_dir, "data", "malaria", "cell_images", "cell_images")
    return create_data_generators(data_dir, batch_size=batch_size)

def load_tb_data(base_dir, batch_size=32):
    # Note: the TB dataset structure from Kaggle might vary. 
    # Update the inner path depending on how it unzips.
    data_dir = os.path.join(base_dir, "data", "tuberculosis", "TB_Chest_Radiography_Database")
    return create_data_generators(data_dir, batch_size=batch_size)

def load_full_production_dataset(base_dir, dataset_name="malaria", target_size=(224, 224), batch_size=32):
    """
    Loads 100% of the raw images into a single training generator (no validation/test split).
    Used EXCLUSIVELY for final production deployment model training.
    """
    if dataset_name == "malaria":
        data_dir = os.path.join(base_dir, "data", "malaria", "cell_images", "cell_images")
    else:
        data_dir = os.path.join(base_dir, "data", "tuberculosis", "TB_Chest_Radiography_Database")
        
    print(f"Loading 100% of {dataset_name} data for Final Production Deployment from {data_dir}...")
    
    production_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=target_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=True
    )
    
    production_ds = production_ds.map(process_batch, num_parallel_calls=tf.data.AUTOTUNE)
    production_ds = production_ds.map(train_augment, num_parallel_calls=tf.data.AUTOTUNE)
    production_ds = production_ds.prefetch(tf.data.AUTOTUNE)
    
    return production_ds
