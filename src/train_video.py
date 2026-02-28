import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import matplotlib.pyplot as plt
from swin_transformer import build_swin_tiny # Assumes your model code is in swin_transformer.py

# Consolidated Config inside the script for easier access
class Config:
    KAGGLE_ENV = os.path.exists('/kaggle')
    FFPP_FRAMES_ROOT = '/kaggle/input/datasets/muhammadqaiser1921/faceforenscis/ffpp_binary_frames'
    DEEPFAKE_FRAMES_ROOT = '/kaggle/input/datasets/aryansingh16/deepfake-dataset/real_vs_fake/real-vs-fake'
    CHECKPOINT_DIR = '/kaggle/working/models/checkpoints'
    LOG_DIR = '/kaggle/working/results/logs'
    WEIGHTS_DIR = '/kaggle/working/weights'
    IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
    FFPP_LABELS = {'0': 0, '1': 1}
    DEEPFAKE_LABELS = {'real': 0, 'fake': 1}
    epochs = 3
    batch_size = 16
    lr = 1e-4

def _collect_image_paths(split_root, class_map, max_per_class=None):
    paths, labels = [], []
    if not os.path.exists(split_root): return paths, labels
    for class_name, label in class_map.items():
        class_dir = os.path.join(split_root, class_name)
        if not os.path.exists(class_dir): continue
        collected = 0
        for name in sorted(os.listdir(class_dir)):
            if name.lower().endswith(Config.IMAGE_EXTS):
                paths.append(os.path.join(class_dir, name))
                labels.append(label)
                collected += 1
                if max_per_class and collected >= max_per_class: break
    return paths, labels

def load_and_prepare_data(max_images=None):
    """Call this ONCE in Kaggle to load paths into memory."""
    print("📂 Loading dataset paths...")
    output = {'train_paths': [], 'train_labels': [], 'val_paths': [], 'val_labels': []}
    datasets = [
        {'root': Config.FFPP_FRAMES_ROOT, 'splits': {'train': 'train', 'val': 'val'}, 'labels': Config.FFPP_LABELS},
        {'root': Config.DEEPFAKE_FRAMES_ROOT, 'splits': {'train': 'train', 'val': 'valid'}, 'labels': Config.DEEPFAKE_LABELS}
    ]
    for d in datasets:
        for sk, folder in d['splits'].items():
            p, l = _collect_image_paths(os.path.join(d['root'], folder), d['labels'], max_images)
            output[f"{sk}_paths"].extend(p)
            output[f"{sk}_labels"].extend(l)
    return output

def run_training_session(data, epochs=Config.epochs, batch_size=Config.batch_size, lr=Config.lr):
    """Main function to call in Kaggle for training."""
    for d in [Config.CHECKPOINT_DIR, Config.LOG_DIR, Config.WEIGHTS_DIR]: os.makedirs(d, exist_ok=True)

    def _decode(p, l):
        img = tf.image.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        return tf.cast(tf.image.resize(img, (224, 224)), tf.float32) / 255.0, l

    train_ds = tf.data.Dataset.from_tensor_slices((data['train_paths'], data['train_labels'])).shuffle(10000).map(_decode).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((data['val_paths'], data['val_labels'])).map(_decode).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Build and compile model
    model = build_swin_tiny(input_shape=(224, 224, 3), num_classes=2)
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=lr),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks = [tf.keras.callbacks.ModelCheckpoint(os.path.join(Config.CHECKPOINT_DIR, f"best_model_{timestamp}.h5"), monitor='val_auc', save_best_only=True, mode='max')]
    
    print(f"🚀 Training for {epochs} epochs...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    return model, history