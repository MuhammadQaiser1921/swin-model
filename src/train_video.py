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
    # Directory to save model checkpoints
    CHECKPOINT_DIR = '/kaggle/working/models/checkpoints'

    # Training hyperparameters
    epochs = 3
    batch_size = 16
    lr = 1e-4

    # Model architecture configuration
    mlp_activation = 'mish'
    attention_activation = 'softmax'
    output_activation = 'sigmoid'
    classifier_hidden_dims = (512, 128)
    classifier_dropout = 0.2


# =========================
# EVALUATION METRICS
# =========================
def _evaluate_binary_threshold(model, dataset, threshold=0.5):
    """
    Compute classification metrics using a fixed probability threshold.
    This is useful for binary classification evaluation beyond default metrics.
    """

    # Collect all ground-truth labels from dataset
    y_true_batches = []
    for _, labels in dataset:
        y_true_batches.append(tf.reshape(labels, [-1]))

    if not y_true_batches:
        return None

    # Concatenate all batches into a single vector
    y_true = tf.concat(y_true_batches, axis=0).numpy().astype(np.int32)

    # Get model predicted probabilities
    y_prob = model.predict(dataset, verbose=0).reshape(-1)

    # Apply threshold to convert probabilities → class predictions
    y_pred = (y_prob >= threshold).astype(np.int32)

    # Compute confusion matrix
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=2).numpy()
    tn, fp, fn, tp = cm.ravel()

    # Derived metrics
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / max(len(y_true), 1)

    # Print metrics
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


# =========================
# MODEL CHECKPOINT SAVING
# =========================
def _save_model_as_pth(model, file_path):
    """
    Save TensorFlow model in a PyTorch-like .pth format.
    Stores both architecture (JSON) and weights.
    """
    payload = {
        'model_config_json': model.to_json(),
        'weights': model.get_weights()
    }
    with open(file_path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


class BestPthCheckpoint(tf.keras.callbacks.Callback):
    """
    Custom callback to save best model based on validation metric.
    """

    def __init__(self, checkpoint_dir, timestamp, monitor='val_accuracy', mode='max'):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.file_path = os.path.join(
            checkpoint_dir,
            f'video_best_model_{timestamp}.pth'
        )

        # Initialize best score depending on optimization direction
        self.best = -np.inf if mode == 'max' else np.inf

    def _is_better(self, current):
        return current > self.best if self.mode == 'max' else current < self.best

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            return

        # Save model if performance improved
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

    # Ensure checkpoint directory exists
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # -------- SANITY CHECK --------
    # Confirms that data is preloaded tensors (Case 2)
    print("🔍 Data Sanity Check:")
    print("Type:", type(data['train_paths'][0]))
    print("Shape:", np.shape(data['train_paths'][0]))

    # -------- NORMALIZATION FUNCTION --------
    # Converts image tensors to float32 and scales to [0, 1]
    def normalize(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        y = tf.cast(y, tf.float32)
        return x, y

    # -------- tf.data PIPELINE --------
    # Since data is already loaded as tensors:
    # → No decoding
    # → No file I/O
    # → Only normalization + batching

    train_ds = (
        tf.data.Dataset
        .from_tensor_slices((data['train_paths'], data['train_labels']))
        .shuffle(10000)
        .map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset
        .from_tensor_slices((data['val_paths'], data['val_labels']))
        .map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Optional test dataset
    test_ds = None
    if data.get('test_paths'):
        test_ds = (
            tf.data.Dataset
            .from_tensor_slices((data['test_paths'], data['test_labels']))
            .map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    # ==============================
    # BUILD MODEL
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

    # -------- COMPILE MODEL --------
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
    )

    # Timestamp for checkpoint naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Callbacks
    callbacks = [
        BestPthCheckpoint(
            checkpoint_dir=Config.CHECKPOINT_DIR,
            timestamp=timestamp,
            monitor='val_accuracy',
            mode='max'
        )
    ]

    # -------- TRAIN MODEL --------
    print(f"🚀 Training for {epochs} epochs...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # -------- EVALUATION --------
    test_metrics = None
    if test_ds is not None:
        print('🧪 Evaluating on test split...')
        test_metrics = model.evaluate(test_ds, return_dict=True)

        print(f"Test metrics: {test_metrics}")

        # Additional threshold-based evaluation
        threshold_metrics = _evaluate_binary_threshold(
            model,
            test_ds,
            threshold=0.5
        )

        if threshold_metrics is not None:
            test_metrics['threshold_metrics'] = threshold_metrics

    return model, history, test_metrics