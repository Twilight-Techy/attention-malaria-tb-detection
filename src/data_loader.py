import tensorflow as tf
import os
import shutil
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
    layers.RandomBrightness(factor=0.2, value_range=(0.0, 1.0)),
])

def train_augment(images, labels):
    images = data_augmentation(images, training=True)
    return images, labels

def create_data_generators(data_dir, output_dir, target_size=(224, 224), batch_size=32, ratio=(0.7, 0.15, 0.15)):
    """
    Physically splits the data into Train, Validation, and Test subsets (default 70/15/15).
    `ratio` is given in splitfolders order: (train, val, test).
    Then creates highly optimized tf.data pipelines for each split.
    """
    if not os.path.exists(output_dir):
        print(f"Splitting dataset into Train/Val/Test subsets {ratio} in {output_dir}...")
        if has_duplicates(data_dir):
            grouped_split(data_dir, output_dir, ratio=ratio, seed=42)
        else:
            # Only image files count towards the split sizes (the malaria folders also hold a Thumbs.db)
            formats = list(IMAGE_EXTS) + [ext.upper() for ext in IMAGE_EXTS]
            splitfolders.ratio(data_dir, output=output_dir, seed=42, ratio=ratio, group_prefix=None, formats=formats)
    
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
    train_ds = train_ds.cache(os.path.join(output_dir, "train_cache"))
    train_ds = train_ds.map(train_augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    val_ds = val_ds.map(process_batch, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.cache(os.path.join(output_dir, "val_cache"))
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    test_ds = test_ds.map(process_batch, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.cache(os.path.join(output_dir, "test_cache"))
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds

def split_to_ratio(split):
    """
    Converts a split tag written Train_Test_Val (e.g. "70_20_10") into the
    (train, val, test) ratio tuple expected by splitfolders.
    """
    train, test, val = (int(v) / 100 for v in split.split("_"))
    return (train, val, test)

def get_split_dir(base_dir, dataset_name, split=None):
    """
    Directory holding the physical train/val/test folders (and tf.data caches) of one split.
    """
    prefix = "malaria" if dataset_name == "malaria" else "tb"
    suffix = f"_{split}" if split else ""
    return os.path.join(base_dir, "data", f"{prefix}_split{suffix}")

def cleanup_split(base_dir, dataset_name, split):
    """
    Deletes a finished split's copied images and caches to free disk space before the next split.
    """
    split_dir = get_split_dir(base_dir, dataset_name, split)
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir, ignore_errors=True)

DUPLICATE_TAG = "__dup"
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')

def build_tb_source_dir(base_dir, tb_data_dir, tb_copies=5):
    """
    Builds the TB dataset used for training: the public Kaggle TB database (3,500 Normal +
    700 Tuberculosis images) with every Tuberculosis image repeated `tb_copies` times in total
    (700 x 5 = 3,500), so both classes hold 3,500 images. Extra copies are named
    <name>__dup<k><ext>. Files are symlinked (the split step copies the real file contents),
    so nothing is duplicated on disk here.
    Returns the directory to split, or tb_data_dir unchanged when tb_copies <= 1.
    """
    if tb_copies <= 1:
        return tb_data_dir

    combined_dir = os.path.join(base_dir, "data", "tb_combined")
    for class_name, copies in (("Normal", 1), ("Tuberculosis", tb_copies)):
        src_dir = os.path.join(tb_data_dir, class_name)
        dst_dir = os.path.join(combined_dir, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        for fname in sorted(os.listdir(src_dir)):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            stem, ext = os.path.splitext(fname)
            for k in range(copies):
                dst = os.path.join(dst_dir, fname if k == 0 else f"{stem}{DUPLICATE_TAG}{k}{ext}")
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(os.path.join(src_dir, fname)), dst)
    counts = {c: len(os.listdir(os.path.join(combined_dir, c))) for c in ("Normal", "Tuberculosis")}
    print(f"Combined TB dataset at {combined_dir}: {counts} (each Tuberculosis image x{tb_copies})")
    return combined_dir

def grouped_split(data_dir, output_dir, ratio, seed=42):
    """
    Splits like splitfolders.ratio (same per-class sizes: int(ratio * n) for train and val, the
    rest for test) but keeps an image and all of its __dup copies in the same subset, so no copy
    of a training image ends up in validation or test.
    `ratio` is given in splitfolders order: (train, val, test).
    """
    import random
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        groups = {}
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(IMAGE_EXTS):
                stem, ext = os.path.splitext(fname)
                groups.setdefault(stem.split(DUPLICATE_TAG)[0] + ext, []).append(fname)
        keys = sorted(groups)
        random.Random(seed).shuffle(keys)
        n = sum(len(v) for v in groups.values())
        n_train = int(ratio[0] * n)
        n_val = int(ratio[1] * n)
        filled = {"train": 0, "val": 0, "test": 0}
        for key in keys:
            subset = "train" if filled["train"] < n_train else ("val" if filled["val"] < n_val else "test")
            dst_dir = os.path.join(output_dir, subset, class_name)
            os.makedirs(dst_dir, exist_ok=True)
            for fname in groups[key]:
                shutil.copy2(os.path.join(class_dir, fname), os.path.join(dst_dir, fname))
            filled[subset] += len(groups[key])
        print(f"Grouped split of {class_name}: {filled}")

def has_duplicates(data_dir):
    return any(DUPLICATE_TAG in f for _, _, files in os.walk(data_dir) for f in files)

def load_malaria_data(base_dir, data_dir=None, batch_size=32, split=None):
    if data_dir is None:
        data_dir = os.path.join(base_dir, "data", "malaria", "cell_images", "cell_images")
    output_dir = get_split_dir(base_dir, "malaria", split)
    if split:
        return create_data_generators(data_dir, output_dir, batch_size=batch_size, ratio=split_to_ratio(split))
    return create_data_generators(data_dir, output_dir, batch_size=batch_size)

def load_tb_data(base_dir, data_dir=None, batch_size=32, split=None):
    # Note: the TB dataset structure from Kaggle might vary. 
    # Update the inner path depending on how it unzips or mounts.
    if data_dir is None:
        data_dir = os.path.join(base_dir, "data", "tuberculosis", "TB_Chest_Radiography_Database")
    output_dir = get_split_dir(base_dir, "tb", split)
    if split:
        return create_data_generators(data_dir, output_dir, batch_size=batch_size, ratio=split_to_ratio(split))
    return create_data_generators(data_dir, output_dir, batch_size=batch_size)

def load_full_production_dataset(base_dir, data_dir=None, dataset_name="malaria", target_size=(224, 224), batch_size=32):
    """
    Loads 100% of the raw images into a single training generator (no validation/test split).
    Used EXCLUSIVELY for final production deployment model training.
    """
    if data_dir is None:
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
    # Use parent directory to avoid cluttering the raw data directory with cache files
    production_ds = production_ds.cache(os.path.join(base_dir, "data", f"{dataset_name}_prod_cache"))
    production_ds = production_ds.map(train_augment, num_parallel_calls=tf.data.AUTOTUNE)
    production_ds = production_ds.prefetch(tf.data.AUTOTUNE)
    
    return production_ds
