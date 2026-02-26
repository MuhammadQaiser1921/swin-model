"""
Deepfake frame dataset training script
======================================

This script trains Swin-Tiny on extracted frames from multiple datasets:
- FaceForensics++ extracted frames (train/val/test with numeric class folders)
- Deepfake_Dataset extracted frames (train/valid/test with class name folders)

It loads images directly from the extracted frame folders, builds tf.data pipelines,
and trains with AUC-focused metrics and callbacks.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model_factory import build_swin_tiny
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from datetime import datetime


# ========== CONFIGURATION ==========

class Config:
    """Training configuration parameters"""
    
    # Environment detection
    KAGGLE_ENV = os.path.exists('/kaggle')
    
    # Dataset paths
    if KAGGLE_ENV:
        # Kaggle dataset paths (read-only mounted datasets)
        FFPP_FRAMES_ROOT = (
            '/kaggle/input/datasets/muhammadqaiser1921/faceforenscis/ffpp_binary_frames'
        )
        DEEPFAKE_FRAMES_ROOT = (
            '/kaggle/input/datasets/aryansingh16/deepfake-dataset/real_vs_fake/real-vs-fake'
        )
        # Output directory for Kaggle
        CHECKPOINT_DIR = '/kaggle/working/models/checkpoints'
        LOG_DIR = '/kaggle/working/results/logs'
        WEIGHTS_DIR = '/kaggle/working/weights'
    else:
        # Local paths
        FFPP_FRAMES_ROOT = r'E:\FYP\Dataset\ffpp_binary_frames'
        DEEPFAKE_FRAMES_ROOT = r'E:\FYP\Dataset\real-vs-fake'
        CHECKPOINT_DIR = os.path.join('..', 'models', 'checkpoints')
        LOG_DIR = os.path.join('..', 'results', 'logs')
        WEIGHTS_DIR = os.path.join('..', 'models', 'weights')

    # Split names in each dataset
    FFPP_SPLITS = {'train': 'train', 'val': 'val', 'test': 'test'}
    DEEPFAKE_SPLITS = {'train': 'train', 'val': 'valid', 'test': 'test'}

    # Label maps
    # FaceForensics++ frames: folder names are "0" and "1" (0=real, 1=fake)
    FFPP_LABELS = {'0': 0, '1': 1}
    # Deepfake dataset: folder names are "real" and "fake"
    DEEPFAKE_LABELS = {'real': 0, 'fake': 1}

    IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
    
    # Deepfake folders (label = 0)
    FAKE_FOLDERS = [
        'DeepFakeDetection',
        'Deepfakes',
        'Face2Face',
        'FaceShifter',
        'FaceSwap',
        'NeuralTextures'
    ]
    
    # Real folder (label = 1)
    REAL_FOLDER = 'original'
    
    # Frame extraction settings
    FRAMES_PER_VIDEO = 10  # Unused for extracted frames
    IMG_SIZE = 224  # Input size for Swin-Tiny (224x224)
    FACE_DETECTION = True  # Set to False to use center crops instead
    
    # Training settings
    BATCH_SIZE = 16  # Adjust based on your GPU memory (Kaggle: 12-16)
    EPOCHS = 50
    INITIAL_LR = 1e-4
    VAL_SPLIT = 0.2  # Used only if val folder is missing
    
    # Model settings
    NUM_CLASSES = 2  # Binary: fake (0) or real (1)
    INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    
    # Data augmentation
    AUGMENTATION = True
    
    # Model saving options
    SAVE_WEIGHTS = True  # Save weights separately
    SAVE_FULL_MODEL = True  # Save entire model
    SAVE_HISTORY = True  # Save training history

    # Optional limit per class (None = all)
    MAX_IMAGES_PER_CLASS = None


# ========== FACE DETECTION ==========

def load_face_detector():
    """
    Load OpenCV's pre-trained face detector (Haar Cascade or DNN).
    Returns None if face detection is disabled in config.
    """
    if not Config.FACE_DETECTION:
        return None
    
    try:
        # Try to load DNN face detector (more accurate)
        model_file = "res10_300x300_ssd_iter_140000.caffemodel"
        config_file = "deploy.prototxt"
        
        if os.path.exists(model_file) and os.path.exists(config_file):
            detector = cv2.dnn.readNetFromCaffe(config_file, model_file)
            print("✓ Loaded DNN face detector")
            return detector
    except:
        pass
    
    try:
        # Fallback to Haar Cascade
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("✓ Loaded Haar Cascade face detector")
        return detector
    except:
        print("⚠ Face detector not available, using center crops")
        return None


def detect_and_crop_face(frame, detector=None, margin=0.2):
    """
    Detect face in frame and crop with margin.
    Falls back to center crop if no face detected.
    
    Args:
        frame: Input frame (BGR format)
        detector: Face detector object
        margin: Additional margin around face (0.2 = 20% larger crop)
    
    Returns:
        Cropped face region (BGR format)
    """
    h, w = frame.shape[:2]
    
    if detector is None:
        # Center crop
        size = min(h, w)
        y_start = (h - size) // 2
        x_start = (w - size) // 2
        return frame[y_start:y_start+size, x_start:x_start+size]
    
    # Try face detection
    try:
        # For DNN detector
        if hasattr(detector, 'forward'):
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            detector.setInput(blob)
            detections = detector.forward()
            
            # Get highest confidence detection
            best_conf = 0
            best_box = None
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > best_conf and confidence > 0.5:
                    best_conf = confidence
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    best_box = box.astype("int")
            
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                # Add margin
                fw, fh = x2 - x1, y2 - y1
                x1 = max(0, int(x1 - fw * margin))
                y1 = max(0, int(y1 - fh * margin))
                x2 = min(w, int(x2 + fw * margin))
                y2 = min(h, int(y2 + fh * margin))
                return frame[y1:y2, x1:x2]
        
        # For Haar Cascade
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Get largest face
                face = max(faces, key=lambda x: x[2] * x[3])
                x, y, fw, fh = face
                # Add margin
                x1 = max(0, int(x - fw * margin))
                y1 = max(0, int(y - fh * margin))
                x2 = min(w, int(x + fw * (1 + margin)))
                y2 = min(h, int(y + fh * (1 + margin)))
                return frame[y1:y2, x1:x2]
    except:
        pass
    
    # Fallback to center crop
    size = min(h, w)
    y_start = (h - size) // 2
    x_start = (w - size) // 2
    return frame[y_start:y_start+size, x_start:x_start+size]


# ========== VIDEO PROCESSING ==========

def extract_frames_from_video(video_path, num_frames=10, detector=None):
    """
    Extract uniformly sampled frames from video with face detection/cropping.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        detector: Face detector object
    
    Returns:
        List of preprocessed frames (224x224x3, normalized)
    """
    frames = []
    
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return []
        
        # Sample frame indices uniformly
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue
            
            # Detect and crop face
            face_crop = detect_and_crop_face(frame, detector)
            
            # Resize to target size
            face_crop = cv2.resize(face_crop, (Config.IMG_SIZE, Config.IMG_SIZE))
            
            # Convert BGR to RGB
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            face_crop = face_crop.astype(np.float32) / 255.0
            
            frames.append(face_crop)
        
        cap.release()
    
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return []
    
    return frames


# ========== DATASET LOADING ==========

def load_faceforensics_dataset(detector=None, max_videos_per_class=None):
    """
    Load FaceForensics++ dataset and extract frames.
    
    Args:
        detector: Face detector object
        max_videos_per_class: Limit videos per class (for testing/debugging)
    
    Returns:
        X: Array of frames (N, 224, 224, 3)
        y: Array of labels (N,) - 0 for fake, 1 for real
        video_info: List of dicts with metadata
    """
    X = []
    y = []
    video_info = []
    
    print("\n" + "="*60)
    print("LOADING FACEFORENSICS++ DATASET")
    print("="*60)
    
    # Process fake videos (label = 0)
    for folder in Config.FAKE_FOLDERS:
        folder_path = os.path.join(Config.DATA_ROOT, folder)
        
        if not os.path.exists(folder_path):
            print(f"⚠ Warning: Folder not found: {folder_path}")
            continue
        
        video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
        
        if max_videos_per_class:
            video_files = video_files[:max_videos_per_class]
        
        print(f"\n📁 Processing {folder}: {len(video_files)} videos")
        
        for video_file in tqdm(video_files, desc=f"  {folder}"):
            video_path = os.path.join(folder_path, video_file)
            frames = extract_frames_from_video(video_path, Config.FRAMES_PER_VIDEO, detector)
            
            if len(frames) > 0:
                X.extend(frames)
                y.extend([0] * len(frames))  # Label 0 = fake
                
                for i in range(len(frames)):
                    video_info.append({
                        'video_path': video_path,
                        'video_name': video_file,
                        'manipulation': folder,
                        'frame_idx': i,
                        'label': 0
                    })
    
    # Process real videos (label = 1)
    real_folder_path = os.path.join(Config.DATA_ROOT, Config.REAL_FOLDER)
    
    if os.path.exists(real_folder_path):
        video_files = [f for f in os.listdir(real_folder_path) if f.endswith('.mp4')]
        
        if max_videos_per_class:
            video_files = video_files[:max_videos_per_class]
        
        print(f"\n📁 Processing {Config.REAL_FOLDER}: {len(video_files)} videos")
        
        for video_file in tqdm(video_files, desc=f"  {Config.REAL_FOLDER}"):
            video_path = os.path.join(real_folder_path, video_file)
            frames = extract_frames_from_video(video_path, Config.FRAMES_PER_VIDEO, detector)
            
            if len(frames) > 0:
                X.extend(frames)
                y.extend([1] * len(frames))  # Label 1 = real
                
                for i in range(len(frames)):
                    video_info.append({
                        'video_path': video_path,
                        'video_name': video_file,
                        'manipulation': 'original',
                        'frame_idx': i,
                        'label': 1
                    })
    else:
        print(f"⚠ Warning: Real folder not found: {real_folder_path}")
    
    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print("\n" + "="*60)
    print(f"✓ Dataset loaded successfully!")
    print(f"  Total frames: {len(X)}")
    print(f"  Fake frames: {np.sum(y == 0)}")
    print(f"  Real frames: {np.sum(y == 1)}")
    print(f"  Frame shape: {X.shape}")
    print("="*60 + "\n")
    
    return X, y, video_info


def _collect_image_paths(split_root, class_map, image_exts, max_per_class=None):
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
    Load extracted frame paths from two datasets with separate split conventions.

    Returns:
        dict with train/val/test paths and labels.
    """
    datasets = [
        {
            'root': Config.FFPP_FRAMES_ROOT,
            'splits': Config.FFPP_SPLITS,
            'labels': Config.FFPP_LABELS,
            'name': 'FaceForensics++',
        },
        {
            'root': Config.DEEPFAKE_FRAMES_ROOT,
            'splits': Config.DEEPFAKE_SPLITS,
            'labels': Config.DEEPFAKE_LABELS,
            'name': 'Deepfake_Dataset',
        },
    ]

    output = {
        'train_paths': [],
        'train_labels': [],
        'val_paths': [],
        'val_labels': [],
        'test_paths': [],
        'test_labels': [],
    }

    for dataset in datasets:
        root = dataset['root']
        splits = dataset['splits']
        labels = dataset['labels']
        name = dataset['name']

        print(f"\n📁 Loading extracted frames: {name}")
        print(f"   Root: {root}")

        for split_key, split_folder in splits.items():
            split_root = os.path.join(root, split_folder)
            paths, lbls = _collect_image_paths(
                split_root,
                labels,
                Config.IMAGE_EXTS,
                max_per_class=Config.MAX_IMAGES_PER_CLASS,
            )

            output[f"{split_key}_paths"].extend(paths)
            output[f"{split_key}_labels"].extend(lbls)

            print(
                f"   {split_key}: {len(paths)} images from {split_root}"
            )

    return output


def _decode_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (Config.IMG_SIZE, Config.IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def build_image_dataset(paths, labels, batch_size, shuffle=False, augment=False):
    if len(paths) == 0:
        return None

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(paths), 20000))

    dataset = dataset.map(_decode_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment and Config.AUGMENTATION:
        augmentation = create_augmentation_layer()
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ========== DATA AUGMENTATION ==========

def create_augmentation_layer():
    """
    Create data augmentation pipeline for training.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),  # ±5% rotation
        layers.RandomZoom(0.1),  # ±10% zoom
        layers.RandomContrast(0.1),  # Contrast adjustment
        layers.RandomBrightness(0.1),  # Brightness adjustment
    ], name="data_augmentation")


def compute_auc_metrics(y_true, y_pred_probs):
    """
    Compute AUC and other threshold-independent metrics.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred_probs: Predicted probabilities for class 1 (shape: N,)
    
    Returns:
        Dictionary with AUC, ROC curve, threshold metrics
    """
    from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, precision_recall_curve
    
    # Compute AUC-ROC
    auc_score = roc_auc_score(y_true, y_pred_probs)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (Youden's index)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Compute at optimal threshold
    y_pred_optimal = (y_pred_probs >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()
    
    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    pr_auc = auc(recall, precision)
    
    return {
        'auc_roc': auc_score,
        'pr_auc': pr_auc,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'optimal_threshold': optimal_threshold,
        'optimal_sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Recall/TPR
        'optimal_specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,  # TNR
        'optimal_precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'optimal_f1': 2 * (tp / (tp + fp) * tp / (tp + fn)) / (tp / (tp + fp) + tp / (tp + fn)) if (tp + fp) > 0 and (tp + fn) > 0 else 0
    }


def plot_auc_curve(metrics, output_path):
    """
    Plot AUC-ROC and Precision-Recall curves.
    
    Args:
        metrics: Dictionary with AUC metrics
        output_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    axes[0].plot(metrics['fpr'], metrics['tpr'], 'b-', linewidth=2, 
                 label=f"AUC = {metrics['auc_roc']:.4f}")
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    axes[0].scatter(1 - metrics['optimal_specificity'], metrics['optimal_sensitivity'], 
                   color='red', s=100, marker='*', label='Optimal Threshold', zorder=5)
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # PR Curve
    from sklearn.metrics import precision_recall_curve
    # Already computed in metrics, we'll plot using stored values
    axes[1].text(0.5, 0.5, f"PR-AUC = {metrics['pr_auc']:.4f}\n\nOptimal Threshold = {metrics['optimal_threshold']:.4f}\n\nSensitivity = {metrics['optimal_sensitivity']:.4f}\nSpecificity = {metrics['optimal_specificity']:.4f}\nPrecision = {metrics['optimal_precision']:.4f}\nF1 = {metrics['optimal_f1']:.4f}", 
                ha='center', va='center', fontsize=12, transform=axes[1].transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1].set_title('Performance Metrics', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ AUC curves saved to: {output_path}")


# ========== WEIGHT SAVING ==========

def save_model_weights(model, model_name="swin_tiny_video"):
    """
    Save model weights, full model, and configuration.
    
    Args:
        model: Trained Keras model
        model_name: Base name for model files
    
    Returns:
        Dictionary with save paths
    """
    os.makedirs(Config.WEIGHTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_paths = {}
    
    # Save weights only (.h5)
    weights_path = os.path.join(Config.WEIGHTS_DIR, f"{model_name}_weights_{timestamp}.h5")
    model.save_weights(weights_path)
    save_paths['weights_h5'] = weights_path
    print(f"✓ Weights saved (H5): {weights_path}")
    
    # Save weights as SavedModel format (TensorFlow)
    saved_model_path = os.path.join(Config.WEIGHTS_DIR, f"{model_name}_savedmodel_{timestamp}")
    model.save(saved_model_path, save_format='tf')
    save_paths['saved_model'] = saved_model_path
    print(f"✓ Model saved (SavedModel): {saved_model_path}")
    
    # Save model configuration
    config_path = os.path.join(Config.WEIGHTS_DIR, f"{model_name}_config_{timestamp}.json")
    config = model.get_config()
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    save_paths['config'] = config_path
    print(f"✓ Model config saved: {config_path}")
    
    # Save model summary
    summary_path = os.path.join(Config.WEIGHTS_DIR, f"{model_name}_summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    save_paths['summary'] = summary_path
    print(f"✓ Model summary saved: {summary_path}")
    
    return save_paths


def create_callbacks(model_name="swin_tiny_video"):
    """
    Create training callbacks for monitoring and checkpointing.
    """
    # Create directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    os.makedirs(Config.WEIGHTS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = []
    
    # ModelCheckpoint - save best model
    checkpoint_path = os.path.join(
        Config.CHECKPOINT_DIR, 
        f"{model_name}_best_{timestamp}.h5"
    )
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_auc',  # Monitor AUC instead of accuracy
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
    )
    
    # EarlyStopping - stop if no improvement
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor='val_auc',  # Monitor AUC
            patience=10,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        )
    )
    
    # ReduceLROnPlateau - reduce learning rate when plateaued
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    )
    
    # TensorBoard - visualize training
    log_path = os.path.join(Config.LOG_DIR, f"{model_name}_{timestamp}")
    callbacks.append(
        keras.callbacks.TensorBoard(
            log_dir=log_path,
            histogram_freq=1,
            write_graph=True
        )
    )
    
    # CSVLogger - log metrics to CSV
    csv_path = os.path.join(Config.LOG_DIR, f"{model_name}_{timestamp}.csv")
    callbacks.append(
        keras.callbacks.CSVLogger(csv_path, separator=',', append=False)
    )
    
    return callbacks


# ========== MAIN TRAINING FUNCTION ==========

def train_video_model():
    """
    Main training pipeline for FaceForensics++ video model.
    """
    print("\n" + "="*60)
    print("SWIN-TINY VIDEO MODEL TRAINING")
    print("FaceForensics++ Dataset (C23 Compression)")
    print("="*60)
    
    # Print configuration
    print("\n📋 Configuration:")
    print(f"  FF++ frames root: {Config.FFPP_FRAMES_ROOT}")
    print(f"  Deepfake frames root: {Config.DEEPFAKE_FRAMES_ROOT}")
    print(f"  Frames per video: {Config.FRAMES_PER_VIDEO}")
    print(f"  Image size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Learning rate: {Config.INITIAL_LR}")
    print(f"  Validation split: {Config.VAL_SPLIT}")
    print(f"  Face detection: {Config.FACE_DETECTION}")
    print(f"  Data augmentation: {Config.AUGMENTATION}")
    
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
        return

    # If val split is missing, create it from train split
    if len(val_paths) == 0:
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
    print(f"  Train fake/real: {np.sum(train_labels == 1)}/{np.sum(train_labels == 0)}")
    print(f"  Val fake/real: {np.sum(val_labels == 1)}/{np.sum(val_labels == 0)}")

    # Create TensorFlow datasets
    train_dataset = build_image_dataset(
        train_paths,
        train_labels,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        augment=True,
    )
    val_dataset = build_image_dataset(
        val_paths,
        val_labels,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        augment=False,
    )
    
    # Build model
    print(f"\n🏗️ Building Swin-Tiny model...")
    model = build_swin_tiny(
        input_shape=Config.INPUT_SHAPE,
        num_classes=Config.NUM_CLASSES
    )
    
    # Print model summary
    print("\n📐 Model Architecture:")
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    print(f"\n  Total parameters: {total_params:,}")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=Config.INITIAL_LR,
            weight_decay=0.05  # Weight decay for regularization
        ),
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    print(f"\n✓ Model compiled successfully!")
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Train model
    print(f"\n🚀 Starting training...\n")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=Config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # ===== EVALUATION & AUC COMPUTATION =====
    print(f"\n📊 Computing AUC and other metrics on validation set...")
    
    # Get predictions on validation set
    y_val_pred_probs = model.predict(val_dataset)
    y_val_pred_probs = y_val_pred_probs[:, 1]  # Get probabilities for class 1 (fake)
    
    # Compute AUC metrics
    auc_metrics = compute_auc_metrics(val_labels, y_val_pred_probs)
    
    # Save final model
    final_model_path = os.path.join(
        Config.CHECKPOINT_DIR,
        f"swin_tiny_video_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
    )
    model.save(final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    # Save model weights
    if Config.SAVE_WEIGHTS:
        weights_paths = save_model_weights(model)
    
    # Save training history
    if Config.SAVE_HISTORY:
        history_path = os.path.join(
            Config.LOG_DIR,
            f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(history_path, 'w') as f:
            json.dump(history.history, f, indent=2)
        print(f"✓ Training history saved to: {history_path}")
    
    # Save AUC metrics
    auc_metrics_path = os.path.join(
        Config.LOG_DIR,
        f"auc_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    metrics_to_save = {k: v for k, v in auc_metrics.items() if k not in ['fpr', 'tpr', 'thresholds']}
    metrics_to_save['fpr'] = [float(x) for x in auc_metrics['fpr']]
    metrics_to_save['tpr'] = [float(x) for x in auc_metrics['tpr']]
    metrics_to_save['thresholds'] = [float(x) for x in auc_metrics['thresholds']]
    
    with open(auc_metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"✓ AUC metrics saved to: {auc_metrics_path}")
    
    # Plot AUC curves
    auc_plot_path = os.path.join(
        Config.LOG_DIR,
        f"auc_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    plot_auc_curve(auc_metrics, auc_plot_path)
    
    # Print final metrics
    print("\n" + "="*60)
    print("TRAINING COMPLETED - FINAL METRICS")
    print("="*60)
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print("\n" + "="*60)
    print("AUC METRICS (on Validation Set)")
    print("="*60)
    print(f"AUC-ROC: {auc_metrics['auc_roc']:.4f}")
    print(f"PR-AUC: {auc_metrics['pr_auc']:.4f}")
    print(f"\nOptimal Threshold: {auc_metrics['optimal_threshold']:.4f}")
    print(f"  Sensitivity (TPR): {auc_metrics['optimal_sensitivity']:.4f}")
    print(f"  Specificity (TNR): {auc_metrics['optimal_specificity']:.4f}")
    print(f"  Precision: {auc_metrics['optimal_precision']:.4f}")
    print(f"  F1-Score: {auc_metrics['optimal_f1']:.4f}")
    print("="*60 + "\n")
    
    # Also compute test metrics if we have test data
    if len(test_paths) > 0:
        print("📊 Computing AUC on test set...")
        test_dataset = build_image_dataset(
            test_paths,
            test_labels,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            augment=False,
        )
        y_test_pred_probs = model.predict(test_dataset)
        y_test_pred_probs = y_test_pred_probs[:, 1]
        test_auc_metrics = compute_auc_metrics(test_labels, y_test_pred_probs)
        
        print(f"\nTest Set AUC-ROC: {test_auc_metrics['auc_roc']:.4f}")
        print(f"Test Set PR-AUC: {test_auc_metrics['pr_auc']:.4f}\n")
    else:
        print("📊 Test split not found. Skipping test metrics.")
    
    return model, history, auc_metrics


# ========== ENTRY POINT ==========

if __name__ == "__main__":
    # Set GPU memory growth (prevents TensorFlow from allocating all GPU memory)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(e)
    
    # Detect environment
    if Config.KAGGLE_ENV:
        print("\n🔍 Running on Kaggle")
    else:
        print("\n🔍 Running locally")
    
    # Run training
    model, history, auc_metrics = train_video_model()