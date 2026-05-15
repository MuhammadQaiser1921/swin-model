import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from pathlib import Path
from ForensiCore import build_video_model

MODEL_NAME = 'ForensiCore-Video'
NUM_CLASSES = 2
PATCH_SIZE = (3, 3)
EMBED_DIM = 64
NUM_HEADS = 8
WINDOW_SIZE = 2
SHIFT_SIZE = 1
NUM_MLP = 256
QKV_BIAS = True
DROPOUT_RATE = 0.03
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
BATCH_SIZE = 32
EPOCHS = 3
RANDOM_STATE = 3
AUTOTUNE = tf.data.AUTOTUNE

# Input size control
# - 'fixed': always resize to FIXED_INPUT_SIZE
# - 'auto': scan dataset and choose most common image size
INPUT_SIZE_MODE = 'fixed'
FIXED_INPUT_SIZE = (224, 224)
SIZE_SCAN_MAX_SAMPLES = 3000

FFPP_FRAMES_ROOT = '/kaggle/input/datasets/muhammadqaiser1921/faceforenscis/ffpp_binary_frames'
DEEPFAKE_FRAMES_ROOT = '/kaggle/input/datasets/aryansingh16/deepfake-dataset/real_vs_fake/real-vs-fake'

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')

# Binary mapping
FFPP_LABELS = {'0': 0, '1': 1}
DEEPFAKE_LABELS = {'real': 0, 'fake': 1}


def collect_labeled_paths(root_dir, label_map, image_exts):
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Warning: dataset root not found -> {root_dir}")
        return [], []

    file_paths = []
    labels = []

    for path in root_path.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in image_exts:
            continue

        path_parts_lower = {part.lower() for part in path.parts}
        matched_label = None
        for folder_name, label in label_map.items():
            if folder_name.lower() in path_parts_lower:
                matched_label = label
                break

        if matched_label is not None:
            file_paths.append(str(path))
            labels.append(matched_label)

    return file_paths, labels


def print_binary_counts(header, labels):
    labels = np.asarray(labels)
    real_count = int(np.sum(labels == 0))
    fake_count = int(np.sum(labels == 1))
    print(f"{header}: {real_count} is real and {fake_count} is fake")


def print_stage_banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_run_config(input_shape):
    print_stage_banner('Run Configuration')
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"input_shape: {input_shape}")
    print(f"NUM_CLASSES: {NUM_CLASSES}")
    print(f"PATCH_SIZE: {PATCH_SIZE}")
    print(f"EMBED_DIM: {EMBED_DIM}, NUM_HEADS: {NUM_HEADS}, NUM_MLP: {NUM_MLP}")
    print(f"WINDOW_SIZE: {WINDOW_SIZE}, SHIFT_SIZE: {SHIFT_SIZE}")
    print(f"DROPOUT_RATE: {DROPOUT_RATE}, LEARNING_RATE: {LEARNING_RATE}, WEIGHT_DECAY: {WEIGHT_DECAY}")
    print(f"TRAIN/VAL/TEST SPLIT: {TRAIN_SPLIT}/{VALIDATION_SPLIT}/{TEST_SPLIT}")
    print(f"BATCH_SIZE: {BATCH_SIZE}, EPOCHS: {EPOCHS}, RANDOM_STATE: {RANDOM_STATE}")
    print(f"INPUT_SIZE_MODE: {INPUT_SIZE_MODE}, FIXED_INPUT_SIZE: {FIXED_INPUT_SIZE}")


def scan_image_sizes(paths, max_samples=3000, seed=3):
    if len(paths) == 0:
        return Counter()

    rng = np.random.default_rng(seed)
    sample_count = min(max_samples, len(paths))
    sampled_idx = rng.choice(len(paths), size=sample_count, replace=False)

    size_counter = Counter()
    for idx in sampled_idx:
        try:
            with load_img(paths[idx]) as image:
                width, height = image.size
            size_counter[(height, width)] += 1
        except Exception:
            continue

    return size_counter


def resolve_input_shape(paths):
    size_counter = scan_image_sizes(paths, max_samples=SIZE_SCAN_MAX_SAMPLES, seed=RANDOM_STATE)
    print_stage_banner('Dataset Image Size Scan')
    if not size_counter:
        print('No image sizes could be scanned. Falling back to FIXED_INPUT_SIZE.')
        return (FIXED_INPUT_SIZE[0], FIXED_INPUT_SIZE[1], 3)

    top_sizes = size_counter.most_common(5)
    print('Top image sizes (height, width) from sampled files:')
    for (h, w), count in top_sizes:
        print(f"- {(h, w)} -> {count} samples")

    most_common_size = top_sizes[0][0]
    if INPUT_SIZE_MODE.lower() == 'auto':
        chosen_size = most_common_size
        print(f"Auto mode: selected most common size {chosen_size}")
    else:
        chosen_size = FIXED_INPUT_SIZE
        print(f"Fixed mode: forcing resize to {chosen_size}")
        if chosen_size != most_common_size:
            print(f"Note: most common dataset size is {most_common_size}, but fixed size is {chosen_size}")

    return (chosen_size[0], chosen_size[1], 3)


def create_balanced_indices(labels, seed=3):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    real_idx = np.where(labels == 0)[0]
    fake_idx = np.where(labels == 1)[0]
    min_count = min(len(real_idx), len(fake_idx))

    if min_count == 0:
        raise ValueError('At least one class has 0 samples. Check dataset paths/mappings.')

    real_bal = rng.choice(real_idx, size=min_count, replace=False)
    fake_bal = rng.choice(fake_idx, size=min_count, replace=False)
    balanced_idx = np.concatenate([real_bal, fake_bal])
    rng.shuffle(balanced_idx)
    return balanced_idx, min_count


def split_balanced(paths, labels, train_split=0.7, val_split=0.15, seed=3):
    labels = np.asarray(labels)
    if not np.isclose(train_split + val_split + TEST_SPLIT, 1.0):
        raise ValueError('Train/validation/test splits must sum to 1.0')

    rng = np.random.default_rng(seed)
    real_idx = np.where(labels == 0)[0]
    fake_idx = np.where(labels == 1)[0]
    per_class_count = min(len(real_idx), len(fake_idx))

    if per_class_count == 0:
        raise ValueError('Cannot split data because one class is empty.')

    real_idx = rng.choice(real_idx, size=per_class_count, replace=False)
    fake_idx = rng.choice(fake_idx, size=per_class_count, replace=False)

    rng.shuffle(real_idx)
    rng.shuffle(fake_idx)

    n_train = int(per_class_count * train_split)
    n_val = int(per_class_count * val_split)
    n_test = per_class_count - n_train - n_val

    def class_split(class_indices):
        train_part = class_indices[:n_train]
        val_part = class_indices[n_train:n_train + n_val]
        test_part = class_indices[n_train + n_val:n_train + n_val + n_test]
        return train_part, val_part, test_part

    real_train, real_val, real_test = class_split(real_idx)
    fake_train, fake_val, fake_test = class_split(fake_idx)

    train_idx = np.concatenate([real_train, fake_train])
    val_idx = np.concatenate([real_val, fake_val])
    test_idx = np.concatenate([real_test, fake_test])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    paths = np.asarray(paths)
    return (
        paths[train_idx].tolist(), labels[train_idx],
        paths[val_idx].tolist(), labels[val_idx],
        paths[test_idx].tolist(), labels[test_idx],
    )


def build_tf_dataset(paths, labels, target_shape, batch_size, shuffle=False, seed=3):
    target_height, target_width = target_shape[:2]
    paths = np.asarray(paths, dtype=str)
    labels = np.asarray(labels, dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def _decode_and_preprocess(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, [target_height, target_width])
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32)
        return image, label

    dataset = dataset.map(_decode_and_preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def run_video_pipeline():
    ffpp_paths, ffpp_labels = collect_labeled_paths(FFPP_FRAMES_ROOT, FFPP_LABELS, IMAGE_EXTS)
    deepfake_paths, deepfake_labels = collect_labeled_paths(DEEPFAKE_FRAMES_ROOT, DEEPFAKE_LABELS, IMAGE_EXTS)

    all_paths = ffpp_paths + deepfake_paths
    all_labels = np.asarray(ffpp_labels + deepfake_labels)

    if len(all_paths) == 0:
        raise ValueError('No images found. Check dataset roots and folder mappings.')

    input_shape = resolve_input_shape(all_paths)
    print_run_config(input_shape)

    print_binary_counts('Collected (before balancing)', all_labels)

    balanced_idx, per_class_balanced = create_balanced_indices(all_labels, seed=RANDOM_STATE)
    all_paths = list(np.asarray(all_paths)[balanced_idx])
    all_labels = all_labels[balanced_idx]

    print_binary_counts('Used after global balancing', all_labels)
    print(f"Balanced per-class sample count: {per_class_balanced}")

    (
        train_paths, y_train_labels,
        val_paths, y_val_labels,
        test_paths, y_test_labels,
    ) = split_balanced(
        all_paths,
        all_labels,
        train_split=TRAIN_SPLIT,
        val_split=VALIDATION_SPLIT,
        seed=RANDOM_STATE,
    )

    print_binary_counts('Train split', y_train_labels)
    print_binary_counts('Validation split', y_val_labels)
    print_binary_counts('Test split', y_test_labels)
    print(f"Total samples used: {len(all_labels)}")
    print(f"Split totals -> train: {len(train_paths)}, val: {len(val_paths)}, test: {len(test_paths)}")

    print_stage_banner('Building tf.data Pipelines')
    train_ds = build_tf_dataset(
        train_paths,
        y_train_labels,
        input_shape,
        BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_STATE,
    )
    val_ds = build_tf_dataset(val_paths, y_val_labels, input_shape, BATCH_SIZE)
    test_ds = build_tf_dataset(test_paths, y_test_labels, input_shape, BATCH_SIZE)

    train_batches = (len(train_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    val_batches = (len(val_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    test_batches = (len(test_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Streaming dataset configured with input_shape: {input_shape}")
    print(f"Train batches: {train_batches}, Validation batches: {val_batches}, Test batches: {test_batches}")

    model = build_video_model(
        input_shape=input_shape,
        num_classes=NUM_CLASSES,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        window_size=WINDOW_SIZE,
        shift_size=SHIFT_SIZE,
        num_mlp=NUM_MLP,
        qkv_bias=QKV_BIAS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        label_smoothing=LABEL_SMOOTHING,
    )

    model.summary()

    early_stopping = EarlyStopping(monitor='val_accuracy',
                                   mode='max',
                                   patience=5,
                                   verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(
        f"{MODEL_NAME}-best.keras",
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    )

    print_stage_banner('Training Started')
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
    )

    best_epoch = int(np.argmax(history.history['val_accuracy']) + 1)
    best_val_acc = float(np.max(history.history['val_accuracy']))
    best_val_loss = float(np.min(history.history['val_loss']))
    print_stage_banner('Training Summary')
    print(f"Best epoch: {best_epoch}")
    print(f"Best val_accuracy: {best_val_acc:.4f}")
    print(f"Best val_loss: {best_val_loss:.4f}")

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    model.save(MODEL_NAME)
    print(f"model saved : {MODEL_NAME}")

    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.savefig(f'{MODEL_NAME}-acc.png')

    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.savefig(f'{MODEL_NAME}-loss.png')


def main():
    run_video_pipeline()


if __name__ == '__main__':
    main()


