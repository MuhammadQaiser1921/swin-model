"""
Verification Script - Check if everything is ready for training
================================================================

This script verifies:
- Python packages installed
- Model factory loads correctly
- Training script has no syntax errors
- Output directories can be created
- TensorFlow/GPU availability

Usage:
    python verify_setup.py
"""

import sys
import os


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(text)
    print("="*60)


def check_python_version():
    """Check Python version"""
    print("\n[1/8] Python Version")
    print(f"  Version: {sys.version}")
    version_info = sys.version_info
    if version_info.major >= 3 and version_info.minor >= 7:
        print("  ✓ Python version OK")
        return True
    else:
        print("  ✗ Python 3.7+ required")
        return False


def check_packages():
    """Check if required packages are installed"""
    print("\n[2/8] Required Packages")
    required = {
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib',
        'pandas': 'Pandas',
        'tqdm': 'tqdm',
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} NOT installed")
            all_ok = False
    
    if not all_ok:
        print("\n  Install with: pip install -r requirements.txt")
    
    return all_ok


def check_tensorflow():
    """Check TensorFlow and GPU"""
    print("\n[3/8] TensorFlow")
    try:
        import tensorflow as tf
        print(f"  Version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  ✓ {len(gpus)} GPU(s) detected:")
            for gpu in gpus:
                print(f"    - {gpu.name}")
        else:
            print("  ⚠ No GPU detected (training will be slow)")
        
        # Test session
        test_tensor = tf.constant([1, 2, 3])
        result = tf.reduce_sum(test_tensor).numpy()
        if result == 6:
            print("  ✓ TensorFlow working correctly")
            return True
    except Exception as e:
        print(f"  ✗ TensorFlow error: {e}")
        return False
    
    return True


def check_model_factory():
    """Check if model_factory.py is valid"""
    print("\n[4/8] Model Factory")
    try:
        from model_factory import build_swin_tiny
        print("  ✓ model_factory.py imports successfully")
        
        # Try building model
        model = build_swin_tiny(input_shape=(224, 224, 3), num_classes=2)
        params = model.count_parameters()
        print(f"  ✓ Model builds successfully (~{params/1e6:.1f}M parameters)")
        return True
    except ImportError as e:
        print(f"  ✗ Cannot import model_factory: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error building model: {e}")
        return False


def check_train_script():
    """Check if train_video.py has syntax errors"""
    print("\n[5/8] Training Script")
    try:
        import py_compile
        py_compile.compile('train_video.py', doraise=True)
        print("  ✓ train_video.py has no syntax errors")
        
        # Try importing Config
        from train_video import Config
        print("  ✓ Config class loads successfully")
        
        # Check config values
        print(f"  - EPOCHS: {Config.EPOCHS}")
        print(f"  - BATCH_SIZE: {Config.BATCH_SIZE}")
        print(f"  - FRAMES_PER_VIDEO: {Config.FRAMES_PER_VIDEO}")
        
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error in train_video.py: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error loading train_video.py: {e}")
        return False


def check_load_script():
    """Check if load_model.py is valid"""
    print("\n[6/8] Model Loading Script")
    try:
        import py_compile
        py_compile.compile('load_model.py', doraise=True)
        print("  ✓ load_model.py has no syntax errors")
        return True
    except Exception as e:
        print(f"  ✗ Error in load_model.py: {e}")
        return False


def check_directories():
    """Check if output directories can be created"""
    print("\n[7/8] Output Directories")
    dirs = [
        '../models/checkpoints',
        '../models/weights',
        '../results/logs',
        '../results/outputs',
    ]
    
    all_ok = True
    for d in dirs:
        try:
            os.makedirs(d, exist_ok=True)
            print(f"  ✓ {d}")
        except Exception as e:
            print(f"  ✗ Cannot create {d}: {e}")
            all_ok = False
    
    return all_ok


def check_dataset_path():
    """Check if dataset path is configured"""
    print("\n[8/8] Dataset Configuration")
    try:
        from train_video import Config
        print(f"  Dataset path: {Config.DATA_ROOT}")
        
        if os.path.exists(Config.DATA_ROOT):
            print("  ✓ Dataset path exists")
            
            # Check for expected folders
            expected = ['original', 'Deepfakes', 'DeepFakeDetection']
            found = []
            for folder in expected:
                if os.path.exists(os.path.join(Config.DATA_ROOT, folder)):
                    found.append(folder)
            
            if found:
                print(f"  ✓ Found {len(found)} expected folders: {', '.join(found)}")
                return True
            else:
                print("  ⚠ FaceForensics++ folders not found yet")
                print("  (You'll need to download the dataset)")
                return True  # Not a critical error
        else:
            print(f"  ⚠ Dataset path doesn't exist: {Config.DATA_ROOT}")
            print("  (Update the path in train_video.py)")
            return True  # Not critical, user can fix
    except Exception as e:
        print(f"  ✗ Error checking dataset: {e}")
        return True  # Not critical


def main():
    print_header("SETUP VERIFICATION SCRIPT")
    print("This will check if your environment is ready for training")
    
    results = {
        'Python': check_python_version(),
        'Packages': check_packages(),
        'TensorFlow': check_tensorflow(),
        'Model Factory': check_model_factory(),
        'Training Script': check_train_script(),
        'Load Script': check_load_script(),
        'Directories': check_directories(),
        'Dataset': check_dataset_path(),
    }
    
    print_header("VERIFICATION SUMMARY")
    
    all_checks = len(results)
    passed_checks = sum(1 for v in results.values() if v)
    
    print(f"\nPassed: {passed_checks}/{all_checks} checks\n")
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print_header("NEXT STEPS")
    
    if all([results['Python'], results['Packages'], 
            results['TensorFlow'], results['Model Factory'],
            results['Training Script']]):
        print("\n✓ Your setup is ready for training!")
        print("\nTo start training:")
        print("  1. Update dataset path in train_video.py (if needed)")
        print("  2. Run: python train_video.py")
        print("\nFor Kaggle:")
        print("  1. Create new Kaggle notebook")
        print("  2. See KAGGLE_GUIDE.md for instructions")
        return 0
    else:
        print("\n✗ Please fix the failed checks before training")
        print("\nTo install packages:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    # Make sure we're in src directory
    if not os.path.exists('train_video.py'):
        print("Error: Run this script from the src/ directory")
        print("Usage: cd src && python verify_setup.py")
        sys.exit(1)
    
    exit_code = main()
    print("\n" + "="*60 + "\n")
    sys.exit(exit_code)
