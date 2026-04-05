import os
import numpy as np
import tensorflow as tf
import pandas as pd
from swin_transformer import build_swin_tiny

print("SCRIPT STARTED")

# =========================
# CONFIG
# =========================
class Config:

    # ✅ FIXED PATH (most likely correct for Kaggle)
    DATA_ROOT = "/kaggle/input/datasets/bishertello/asvspoof-21-df-cqt/my_dataset"

    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    VAL_DIR = os.path.join(DATA_ROOT, "validation")
    TEST_DIR = os.path.join(DATA_ROOT, "test")

    IMAGE_EXTS = (".jpg", ".jpeg", ".png")

    LABELS = {"real": 0, "fake": 1}

    batch_size = 16
    epochs = 3
    lr = 1e-4

    CHECKPOINT_DIR = "/kaggle/working/models"

    ACTIVATIONS = ["gelu", "swish", "mish", "relu"]


# =========================
# LOAD DATA PATHS
# =========================
def collect_paths(root):

    paths = []
    labels = []

    if not os.path.exists(root):
        print(f"❌ Folder NOT found: {root}")
        return [], []

    for cls in os.listdir(root):

        cls_path = os.path.join(root, cls)

        if not os.path.isdir(cls_path):
            continue

        # ✅ Flexible labeling
        label = 0 if "real" in cls.lower() else 1

        for img in os.listdir(cls_path):

            if img.lower().endswith(Config.IMAGE_EXTS):

                paths.append(os.path.join(cls_path, img))
                labels.append(label)

    print(f"Loaded {len(paths)} samples from {root}")

    # limit for faster testing
    return paths[:1000], labels[:1000]


def load_data():

    print("\n🔍 Checking dataset structure...")
    print("Input dir:", os.listdir("/kaggle/input"))

    print("\nLoading training paths...")
    train_p, train_l = collect_paths(Config.TRAIN_DIR)

    print("\nLoading validation paths...")
    val_p, val_l = collect_paths(Config.VAL_DIR)

    print("\nLoading test paths...")
    test_p, test_l = collect_paths(Config.TEST_DIR)

    print("\n📊 DATA SUMMARY")
    print("Train:", len(train_p))
    print("Val:", len(val_p))
    print("Test:", len(test_p))

    print("\nSample train paths:", train_p[:3])
    print("Sample labels:", train_l[:3])

    return train_p, train_l, val_p, val_l, test_p, test_l


# =========================
# IMAGE DECODER
# =========================
def decode(path, label):

    img = tf.io.read_file(path)

    # ✅ FIX: supports PNG + JPG
    img = tf.image.decode_image(img, channels=3)

    # ✅ REQUIRED after decode_image
    img.set_shape([None, None, 3])

    img = tf.image.resize(img, (224, 224))

    img = tf.cast(img, tf.float32) / 255.0

    # ✅ FIX: proper shape for BCE
    label = tf.cast(label, tf.float32)
    label = tf.expand_dims(label, axis=-1)

    return img, label


# =========================
# DATASET BUILDER
# =========================
def make_dataset(paths, labels, shuffle=False):

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(10000)

    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(Config.batch_size)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# =========================
# METRICS
# =========================
def evaluate_metrics(model, ds):

    y_true = []
    y_pred = []

    for x, y in ds:

        preds = model.predict(x, verbose=0)

        y_true.extend(y.numpy().flatten())
        y_pred.extend((preds > 0.5).astype(int).flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    acc = (tp + tn) / len(y_true)

    return acc, precision, recall, f1


# =========================
# TRAIN EXPERIMENTS
# =========================
def run_experiments():

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    train_p, train_l, val_p, val_l, test_p, test_l = load_data()

    # ❌ STOP if dataset not loaded
    if len(train_p) == 0:
        print("\n❌ ERROR: No training data found. Check dataset path.")
        return

    train_ds = make_dataset(train_p, train_l, shuffle=True)
    val_ds = make_dataset(val_p, val_l)
    test_ds = make_dataset(test_p, test_l)

    results = []

    for act in Config.ACTIVATIONS:

        print("\n==========================")
        print("🚀 Training with:", act)
        print("==========================")

        model = build_swin_tiny(
            input_shape=(224, 224, 3),
            num_classes=1,
            activation=act
        )

        model.compile(
            optimizer=tf.keras.optimizers.AdamW(Config.lr),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        print("Starting training...")

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=Config.epochs
        )

        acc, prec, rec, f1 = evaluate_metrics(model, test_ds)

        print("\n✅ RESULTS")
        print("Accuracy:", acc)
        print("Precision:", prec)
        print("Recall:", rec)
        print("F1:", f1)

        save_path = os.path.join(
            Config.CHECKPOINT_DIR,
            f"swin_audio_{act}.h5"
        )

        model.save(save_path)

        results.append({
            "Activation": act,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
        })

    df = pd.DataFrame(results)

    print("\n📊 FINAL RESULTS TABLE")
    print(df)

    df.to_csv("/kaggle/working/audio_results.csv", index=False)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("🚀 FORCE RUN START", flush=True)
    run_experiments()
