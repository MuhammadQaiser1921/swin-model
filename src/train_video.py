import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from swin_transformer import build_swin_tiny


# =========================
# CONFIG
# =========================
class Config:
    FFPP_FRAMES_ROOT = '/kaggle/input/datasets/muhammadqaiser1921/faceforenscis/ffpp_binary_frames'
    DEEPFAKE_FRAMES_ROOT = '/kaggle/input/datasets/aryansingh16/deepfake-dataset/real_vs_fake/real-vs-fake'

    CHECKPOINT_DIR = '/kaggle/working/models/checkpoints'
    IMAGE_EXTS = ('.jpg', '.jpeg', '.png')

    # Binary mapping
    FFPP_LABELS = {'0': 0, '1': 1}
    DEEPFAKE_LABELS = {'real': 0, 'fake': 1}

    epochs = 3
    batch_size = 16
    lr = 1e-4


# =========================
# DATA LOADER
# =========================
def _collect_image_paths(split_root, class_map):
    paths, labels = [], []

    if not os.path.exists(split_root):
        return paths, labels

    for class_name, label in class_map.items():
        class_dir = os.path.join(split_root, class_name)
        if not os.path.exists(class_dir):
            continue

        for name in os.listdir(class_dir):
            if name.lower().endswith(Config.IMAGE_EXTS):
                paths.append(os.path.join(class_dir, name))
                labels.append(label)

    return paths, labels


def load_and_prepare_data():
    print("📂 Loading dataset paths...")

    output = {
        'train_paths': [],
        'train_labels': [],
        'val_paths': [],
        'val_labels': []
    }

    datasets = [
        {
            'root': Config.FFPP_FRAMES_ROOT,
            'splits': {'train': 'train', 'val': 'val'},
            'labels': Config.FFPP_LABELS
        },
        {
            'root': Config.DEEPFAKE_FRAMES_ROOT,
            'splits': {'train': 'train', 'val': 'valid'},
            'labels': Config.DEEPFAKE_LABELS
        }
    ]

    for d in datasets:
        for split_key, folder in d['splits'].items():
            paths, labels = _collect_image_paths(
                os.path.join(d['root'], folder),
                d['labels']
            )

            output[f"{split_key}_paths"].extend(paths)
            output[f"{split_key}_labels"].extend(labels)

    print(f"Train samples: {len(output['train_paths'])}")
    print(f"Val samples: {len(output['val_paths'])}")

    return output


# =========================
# TRAINING FUNCTION
# =========================
def run_training_session(
        data,
        epochs=Config.epochs,
        batch_size=Config.batch_size,
        lr=Config.lr):

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # -------- Decode Function --------
    def _decode(path, label):
        img = tf.image.decode_image(
            tf.io.read_file(path),
            channels=3,
            expand_animations=False
        )
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    # -------- tf.data Pipeline --------
    train_ds = (
        tf.data.Dataset
        .from_tensor_slices((data['train_paths'], data['train_labels']))
        .shuffle(10000)
        .map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset
        .from_tensor_slices((data['val_paths'], data['val_labels']))
        .map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ==============================
    # BUILD MODEL ON GPU (CRITICAL FIX)
    # ==============================
    
    model = build_swin_tiny(
            input_shape=(224, 224, 3),
            num_classes=2
        )

    # -------- Compile --------
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy'
        ]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(
                Config.CHECKPOINT_DIR,
                f"best_model_{timestamp}.h5"
            ),
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        )
    ]

    print(f"🚀 Training for {epochs} epochs...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    return model, history