"""
Data Loading and Preparation for Deepfake Detection Training
===========================================================

This module contains all functions for:
- Collecting image paths from the dataset folders.
- Creating and splitting train/validation/test sets.
- Building `tf.data.Dataset` objects for training.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from video_train_config import Config


def _collect_image_paths(split_root, class_map, image_exts, max_per_class=None):
    """
    Recursively find all image paths in a directory for given classes.
    
    Args:
        split_root (str): Path to the split folder (e.g., 'train', 'val').
        class_map (dict): Maps class folder names to integer labels.
        image_exts (tuple): Tuple of valid image extensions (e.g., '.jpg', '.png').
        max_per_class (int, optional): Max images to load per class. Defaults to None.

    Returns:
        tuple: (list of paths, list of labels)
    """
    paths = []
    labels = []

    if not os.path.exists(split_root):
        return paths, labels

    for class_name, label in class_map.items():
        class_dir = os.path.join(split_root, class_name)
        if not os.path.exists(class_dir):
            continue

        collected = 0
        for root, _, files in os.walk(class_dir):
            for name in sorted(files):
                if not name.lower().endswith(image_exts):
                    continue
                paths.append(os.path.join(root, name))
                labels.append(label)
                collected += 1
                if max_per_class and collected >= max_per_class:
                    break
            if max_per_class and collected >= max_per_class:
                break

    return paths, labels


def load_extracted_frame_paths():
    """
    Load image paths and labels from both FaceForensics++ and Deepfake datasets.
    
    Returns:
        dict: Contains train, validation, and test paths and labels.
    """
    all_paths = {'train': [], 'val': [], 'test': []}
    all_labels = {'train': [], 'val': [], 'test': []}

    datasets = {
        'ffpp': {
            'root': Config.FFPP_FRAMES_ROOT,
            'splits': Config.FFPP_SPLITS,
            'labels': Config.FFPP_LABELS,
        },
        'deepfake': {
            'root': Config.DEEPFAKE_FRAMES_ROOT,
            'splits': Config.DEEPFAKE_SPLITS,
            'labels': Config.DEEPFAKE_LABELS,
        },
    }

    print("\n🔍 Searching for frames in dataset folders...")
    for name, d in datasets.items():
        print(f"  -> Processing '{name}' dataset at: {d['root']}")
        for split_name, split_folder in d['splits'].items():
            split_path = os.path.join(d['root'], split_folder)
            paths, labels = _collect_image_paths(
                split_path, d['labels'], Config.IMAGE_EXTS, Config.MAX_IMAGES_PER_CLASS
            )
            all_paths[split_name].extend(paths)
            all_labels[split_name].extend(labels)
            print(f"    - Found {len(paths)} frames in '{split_folder}' ({split_name} split)")

    return {
        'train_paths': all_paths['train'],
        'train_labels': all_labels['train'],
        'val_paths': all_paths['val'],
        'val_labels': all_labels['val'],
        'test_paths': all_paths['test'],
        'test_labels': all_labels['test'],
    }


def _parse_image(filename, label):
    """Load and decode an image file."""
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [Config.IMG_SIZE, Config.IMG_SIZE])
    return image, label


def _augment_image(image, label):
    """Apply random data augmentation."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def build_image_dataset(paths, labels, batch_size, shuffle=False, augment=False):
    """
    Build a tf.data.Dataset from image paths and labels.
    
    Args:
        paths (list): List of image file paths.
        labels (list): List of corresponding integer labels.
        batch_size (int): The batch size for the dataset.
        shuffle (bool): Whether to shuffle the dataset.
        augment (bool): Whether to apply data augmentation.

    Returns:
        tf.data.Dataset: The configured dataset.
    """
    if not paths:
        return tf.data.Dataset.from_tensor_slices(([], []))

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
        
    dataset = dataset.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        dataset = dataset.map(_augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset


def load_data():
    """
    Load and split data into train/val/test sets.
    Call this once and reuse the returned data.
    
    Returns:
        dict: Contains train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
    """
    print("\n📂 Loading data...")
    
    # Load extracted frame paths
    paths = load_extracted_frame_paths()

    train_paths = paths['train_paths']
    train_labels = np.array(paths['train_labels'], dtype=np.int32)
    val_paths = paths['val_paths']
    val_labels = np.array(paths['val_labels'], dtype=np.int32)
    test_paths = paths['test_paths']
    test_labels = np.array(paths['test_labels'], dtype=np.int32)

    if len(train_paths) == 0:
        print("❌ Error: No training data found. Check dataset roots and split folders.")
        return None

    # If val split is missing, create it from train split
    if len(val_paths) == 0 and Config.VAL_SPLIT > 0:
        print(f"  -> No validation set found. Creating one with {Config.VAL_SPLIT*100}% of training data.")
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths,
            train_labels,
            test_size=Config.VAL_SPLIT,
            random_state=42,
            stratify=train_labels,
        )

    print(f"\n📊 Data split:")
    print(f"  Training samples: {len(train_paths)}")
    print(f"  Validation samples: {len(val_paths)}")
    print(f"  Test samples: {len(test_paths)}")
    print(f"  Train fake/real: {np.sum(train_labels == 1)}/{np.sum(train_labels == 0)}")
    print(f"  Val fake/real: {np.sum(val_labels == 1)}/{np.sum(val_labels == 0)}")
    
    return {
        'train_paths': train_paths,
        'train_labels': train_labels,
        'val_paths': val_paths,
        'val_labels': val_labels,
        'test_paths': test_paths,
        'test_labels': test_labels
    }


def prepare_datasets(data):
    """
    Create TensorFlow datasets from loaded data.
    
    Args:
        data: Dict from load_data() containing paths and labels
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    print("\n🔄 Preparing TensorFlow datasets...")
    
    train_dataset = build_image_dataset(
        data['train_paths'],
        data['train_labels'],
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        augment=True,
    )
    val_dataset = build_image_dataset(
        data['val_paths'],
        data['val_labels'],
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        augment=False,
    )
    
    test_dataset = None
    if len(data['test_paths']) > 0:
        test_dataset = build_image_dataset(
            data['test_paths'],
            data['test_labels'],
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            augment=False,
        )
    
    print("✓ Datasets prepared")
    return train_dataset, val_dataset, test_dataset
