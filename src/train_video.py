import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from swin_transformer import build_video_swin


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
    mlp_activation = 'mish'
    attention_activation = 'softmax'
    output_activation = 'sigmoid'
    classifier_hidden_dims = (512, 128)
    classifier_dropout = 0.2


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
        'val_labels': [],
        'test_paths': [],
        'test_labels': []
    }

    datasets = [
        {
            'root': Config.FFPP_FRAMES_ROOT,
            'splits': {'train': 'train', 'val': 'val', 'test': 'test'},
            'labels': Config.FFPP_LABELS
        },
        {
            'root': Config.DEEPFAKE_FRAMES_ROOT,
            'splits': {'train': 'train', 'val': 'valid', 'test': 'test'},
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
    print(f"Test samples: {len(output['test_paths'])}")

    return output


def _evaluate_binary_threshold(model, dataset, threshold=0.5):
    # Collect true labels from dataset batches.
    y_true_batches = []
    for _, labels in dataset:
        y_true_batches.append(tf.reshape(labels, [-1]))

    if not y_true_batches:
        return None

    y_true = tf.concat(y_true_batches, axis=0).numpy().astype(np.int32)
    y_prob = model.predict(dataset, verbose=0).reshape(-1)
    y_pred = (y_prob >= threshold).astype(np.int32)

    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=2).numpy()
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / max(len(y_true), 1)

    print(f"\n📌 Threshold metrics @ {threshold:.2f}")
    print(f"Confusion Matrix [[TN, FP], [FN, TP]]:\n{cm}")
    print(
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | Accuracy: {accuracy:.4f}"
    )

    return {
        'threshold': float(threshold),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'threshold_accuracy': float(accuracy)
    }


def _save_model_as_pth(model, file_path):
    payload = {
        'model_config_json': model.to_json(),
        'weights': model.get_weights()
    }
    with open(file_path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


class BestPthCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, timestamp, monitor='val_accuracy', mode='max'):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.file_path = os.path.join(checkpoint_dir, f'video_best_model_{timestamp}.pth')
        self.best = -np.inf if mode == 'max' else np.inf

    def _is_better(self, current):
        if self.mode == 'max':
            return current > self.best
        return current < self.best

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if self._is_better(current):
            self.best = current
            _save_model_as_pth(self.model, self.file_path)
            print(
                f"\n💾 Saved best .pth at epoch {epoch + 1} "
                f"({self.monitor}: {current:.4f}) -> {self.file_path}"
            )


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
    path = tf.cast(path, tf.string)  # <-- critical guard

    img = tf.image.decode_image(
        tf.io.read_file(path),
        channels=3,
        expand_animations=False
    )
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32) / 255.0
    label = tf.cast(label, tf.float32)

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

    test_ds = None
    if data.get('test_paths'):
        test_ds = (
            tf.data.Dataset
            .from_tensor_slices((data['test_paths'], data['test_labels']))
            .map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    # ==============================
    # BUILD MODEL ON GPU (CRITICAL FIX)
    # ==============================
    
    model = build_video_swin(
            input_shape=(224, 224, 3),
            num_classes=1,
            mlp_activation=Config.mlp_activation,
            attention_activation=Config.attention_activation,
            output_activation=Config.output_activation,
            classifier_hidden_dims=Config.classifier_hidden_dims,
            classifier_dropout=Config.classifier_dropout
        )

    # -------- Compile --------
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy')
        ]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    callbacks = [
        BestPthCheckpoint(
            checkpoint_dir=Config.CHECKPOINT_DIR,
            timestamp=timestamp,
            monitor='val_accuracy',
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

    test_metrics = None
    if test_ds is not None:
        print('🧪 Evaluating on test split...')
        test_metrics = model.evaluate(test_ds, return_dict=True)
        print(f"Test metrics: {test_metrics}")
        threshold_metrics = _evaluate_binary_threshold(model, test_ds, threshold=0.5)
        if threshold_metrics is not None:
            test_metrics['threshold_metrics'] = threshold_metrics

    return model, history, test_metrics
