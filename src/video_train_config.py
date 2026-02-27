"""
Video Training Configuration
=============================

Configuration parameters for training Swin-Tiny on pre-extracted deepfake frames.
"""

import os


class Config:
    """Training configuration parameters"""
    
    # Environment detection
    KAGGLE_ENV = os.path.exists('/kaggle')
    
    # Dataset paths
    if KAGGLE_ENV:
        FFPP_FRAMES_ROOT = (
            '/kaggle/input/datasets/muhammadqaiser1921/faceforenscis/ffpp_binary_frames'
        )
        DEEPFAKE_FRAMES_ROOT = (
            '/kaggle/input/datasets/aryansingh16/deepfake-dataset/real_vs_fake/real-vs-fake'
        )
        CHECKPOINT_DIR = '/kaggle/working/models/checkpoints'
        LOG_DIR = '/kaggle/working/results/logs'
        WEIGHTS_DIR = '/kaggle/working/weights'
    else:
        FFPP_FRAMES_ROOT = r'E:\FYP\Dataset\ffpp_binary_frames'
        DEEPFAKE_FRAMES_ROOT = r'E:\FYP\Dataset\real-vs-fake'
        CHECKPOINT_DIR = os.path.join('..', 'models', 'checkpoints')
        LOG_DIR = os.path.join('..', 'results', 'logs')
        WEIGHTS_DIR = os.path.join('..', 'models', 'weights')

    # Split names in each dataset
    FFPP_SPLITS = {'train': 'train', 'val': 'val', 'test': 'test'}
    DEEPFAKE_SPLITS = {'train': 'train', 'val': 'valid', 'test': 'test'}

    # Label maps: FaceForensics++ folders are "0" and "1" (0=real, 1=fake)
    FFPP_LABELS = {'0': 0, '1': 1}
    # Deepfake dataset: folder names are "real" and "fake"
    DEEPFAKE_LABELS = {'real': 0, 'fake': 1}

    IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
    
    # Training settings
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 50
    INITIAL_LR = 1e-4
    VAL_SPLIT = 0.2
    
    # Model settings
    NUM_CLASSES = 2
    INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    
    # Data augmentation
    AUGMENTATION = True
    SAVE_WEIGHTS = True
    SAVE_HISTORY = True

    # Optional limit per class (None = all)
    MAX_IMAGES_PER_CLASS = None
