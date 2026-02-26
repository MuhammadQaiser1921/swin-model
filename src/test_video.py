"""
Evaluate a trained Swin-Tiny model on extracted frame datasets and
produce confusion matrix + classification report + AUC metrics.
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    auc,
)
import matplotlib.pyplot as plt

from model_factory import build_swin_tiny


IMAGE_EXTS = (".jpg", ".jpeg", ".png")
FFPP_LABELS = {"0": 0, "1": 1}  # 0=real, 1=fake
DEEPFAKE_LABELS = {"real": 0, "fake": 1}


def _collect_image_paths(split_root, class_map):
    paths = []
    labels = []

    if not os.path.exists(split_root):
        return paths, labels

    for class_name, label in class_map.items():
        class_dir = os.path.join(split_root, class_name)
        if not os.path.exists(class_dir):
            continue

        for root, _, files in os.walk(class_dir):
            for name in sorted(files):
                if not name.lower().endswith(IMAGE_EXTS):
                    continue
                paths.append(os.path.join(root, name))
                labels.append(label)

    return paths, labels


def load_test_paths(ffpp_root, deepfake_root):
    ffpp_test_root = os.path.join(ffpp_root, "test")
    deepfake_test_root = os.path.join(deepfake_root, "test")

    ffpp_paths, ffpp_labels = _collect_image_paths(ffpp_test_root, FFPP_LABELS)
    deep_paths, deep_labels = _collect_image_paths(deepfake_test_root, DEEPFAKE_LABELS)

    paths = ffpp_paths + deep_paths
    labels = ffpp_labels + deep_labels

    return paths, np.array(labels, dtype=np.int32)


def _decode_image(path, label, img_size):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (img_size, img_size))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def build_dataset(paths, labels, batch_size, img_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, l: _decode_image(p, l, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def save_confusion_matrix(cm, output_path, labels=("Real", "Fake")):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on extracted frames")
    parser.add_argument("--ffpp-root", required=True, help="FF++ frames root")
    parser.add_argument("--deepfake-root", required=True, help="Deepfake frames root")
    parser.add_argument("--weights", default=None, help="Path to model weights (.h5)")
    parser.add_argument("--saved-model", default=None, help="Path to SavedModel dir")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="../results/logs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    paths, labels = load_test_paths(args.ffpp_root, args.deepfake_root)
    if len(paths) == 0:
        raise RuntimeError("No test images found. Check test folder paths.")

    dataset = build_dataset(paths, labels, args.batch_size, args.img_size)

    if args.saved_model:
        model = tf.keras.models.load_model(args.saved_model)
    elif args.weights:
        model = build_swin_tiny((args.img_size, args.img_size, 3), 2)
        model.load_weights(args.weights)
    else:
        raise RuntimeError("Provide --weights or --saved-model")

    probs = model.predict(dataset)
    probs = probs[:, 1]
    preds = (probs >= args.threshold).astype(int)

    acc = accuracy_score(labels, preds)
    auc_roc = roc_auc_score(labels, probs)
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)

    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=["Real", "Fake"])

    print("\nAccuracy:", f"{acc:.4f}")
    print("AUC-ROC:", f"{auc_roc:.4f}")
    print("PR-AUC:", f"{pr_auc:.4f}")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_path = os.path.join(args.output_dir, f"confusion_matrix_{timestamp}.png")
    save_confusion_matrix(cm, cm_path)

    metrics_path = os.path.join(args.output_dir, f"test_metrics_{timestamp}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": acc,
                "auc_roc": auc_roc,
                "pr_auc": pr_auc,
                "threshold": args.threshold,
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
            },
            f,
            indent=2,
        )

    print(f"\nSaved confusion matrix: {cm_path}")
    print(f"Saved metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()