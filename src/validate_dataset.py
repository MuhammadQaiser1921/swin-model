"""
Dataset Validation Script for FaceForensics++
==============================================

This script validates your FaceForensics++ dataset structure and provides
a summary before training. Run this before starting training to catch issues early.

Usage:
    python validate_dataset.py
"""

import os
import sys

# Dataset path - update this to match your train_video.py config
DATASET_PATH = r'E:\FYP\Dataset\FaceForensics++'

# Expected folders
EXPECTED_FOLDERS = {
    'fake': [
        'DeepFakeDetection',
        'Deepfakes',
        'Face2Face',
        'FaceShifter',
        'FaceSwap',
        'NeuralTextures'
    ],
    'real': ['original']
}


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def check_path_exists(path):
    """Check if path exists"""
    if os.path.exists(path):
        print(f"✓ Path exists: {path}")
        return True
    else:
        print(f"✗ Path NOT found: {path}")
        return False


def count_videos(folder_path):
    """Count video files in folder"""
    if not os.path.exists(folder_path):
        return 0
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in os.listdir(folder_path) 
                   if os.path.isfile(os.path.join(folder_path, f)) 
                   and os.path.splitext(f)[1].lower() in video_extensions]
    return len(video_files)


def validate_dataset():
    """Validate FaceForensics++ dataset structure"""
    
    print_header("FACEFORENSICS++ DATASET VALIDATION")
    
    # Check main dataset path
    print("\n📁 Checking dataset root path...")
    if not check_path_exists(DATASET_PATH):
        print("\n❌ ERROR: Dataset root path not found!")
        print(f"   Please update DATASET_PATH in this script to point to your dataset.")
        print(f"   Current path: {DATASET_PATH}")
        return False
    
    # Check fake folders
    print("\n📁 Checking deepfake folders (6 manipulation methods)...")
    fake_folders_ok = True
    total_fake_videos = 0
    
    for folder in EXPECTED_FOLDERS['fake']:
        folder_path = os.path.join(DATASET_PATH, folder)
        video_count = count_videos(folder_path)
        
        if video_count > 0:
            print(f"  ✓ {folder}: {video_count} videos")
            total_fake_videos += video_count
        else:
            print(f"  ✗ {folder}: NOT FOUND or EMPTY")
            fake_folders_ok = False
    
    # Check real folder
    print("\n📁 Checking real video folder...")
    real_folders_ok = True
    total_real_videos = 0
    
    for folder in EXPECTED_FOLDERS['real']:
        folder_path = os.path.join(DATASET_PATH, folder)
        video_count = count_videos(folder_path)
        
        if video_count > 0:
            print(f"  ✓ {folder}: {video_count} videos")
            total_real_videos += video_count
        else:
            print(f"  ✗ {folder}: NOT FOUND or EMPTY")
            real_folders_ok = False
    
    # Summary
    print_header("DATASET SUMMARY")
    
    print(f"\n📊 Video Statistics:")
    print(f"  Deepfake videos: {total_fake_videos}")
    print(f"  Real videos: {total_real_videos}")
    print(f"  Total videos: {total_fake_videos + total_real_videos}")
    
    # Expected counts
    print(f"\n📋 Expected (FaceForensics++ C23 full dataset):")
    print(f"  Deepfake videos: 6,000 (1,000 per manipulation)")
    print(f"  Real videos: 1,000")
    print(f"  Total: 7,000 videos")
    
    # Calculate percentages
    if total_fake_videos > 0 or total_real_videos > 0:
        fake_percent = (total_fake_videos / 6000) * 100
        real_percent = (total_real_videos / 1000) * 100
        total_percent = ((total_fake_videos + total_real_videos) / 7000) * 100
        
        print(f"\n📈 Dataset Completeness:")
        print(f"  Deepfake: {fake_percent:.1f}% of expected")
        print(f"  Real: {real_percent:.1f}% of expected")
        print(f"  Overall: {total_percent:.1f}% of expected")
    
    # Validation result
    print_header("VALIDATION RESULT")
    
    if fake_folders_ok and real_folders_ok and (total_fake_videos + total_real_videos) > 0:
        print("\n✓ Dataset validation PASSED!")
        print("\n  Your dataset appears to be correctly structured.")
        print("  You can proceed with training.")
        
        # Training estimates
        print_header("TRAINING ESTIMATES")
        
        total_videos = total_fake_videos + total_real_videos
        frames_per_video = 10  # Default in config
        total_frames = total_videos * frames_per_video
        
        print(f"\n⏱️  Estimated frames to process:")
        print(f"  With {frames_per_video} frames/video: ~{total_frames:,} frames")
        
        print(f"\n💾 Estimated memory requirements:")
        # Each frame: 224x224x3 = 150KB approx, float32 = 4x = 600KB
        memory_gb = (total_frames * 224 * 224 * 3 * 4) / (1024**3)
        print(f"  Peak RAM usage: ~{memory_gb:.1f} GB")
        print(f"  Recommended RAM: {max(16, int(memory_gb * 1.5))} GB+")
        
        print(f"\n🎯 Training recommendations:")
        if total_videos < 1000:
            print("  ⚠️  Small dataset detected")
            print("  - Consider strong data augmentation")
            print("  - Use smaller validation split (0.1-0.15)")
        elif total_videos < 3500:
            print("  📊 Medium dataset")
            print("  - Standard augmentation recommended")
            print("  - Validation split: 0.15-0.2")
        else:
            print("  📊 Large dataset")
            print("  - Moderate augmentation sufficient")
            print("  - Validation split: 0.2")
        
        print("\n🚀 Ready to train!")
        print("   Run: cd src && python train_video.py")
        
        return True
    else:
        print("\n❌ Dataset validation FAILED!")
        print("\n  Issues detected:")
        if not fake_folders_ok:
            print("  - Some or all deepfake folders are missing/empty")
        if not real_folders_ok:
            print("  - Real video folder is missing/empty")
        if total_fake_videos + total_real_videos == 0:
            print("  - No videos found in dataset")
        
        print("\n  Please ensure:")
        print("  1. Dataset is fully downloaded and extracted")
        print("  2. DATASET_PATH points to correct location")
        print("  3. Folder structure matches FaceForensics++ format")
        
        return False


def check_quick_test_sample(num_videos=10):
    """Check if enough videos for quick test"""
    print_header("QUICK TEST VALIDATION")
    
    print(f"\n🔍 Checking if dataset has enough videos for quick test ({num_videos} per class)...")
    
    sufficient = True
    for folder in EXPECTED_FOLDERS['fake']:
        folder_path = os.path.join(DATASET_PATH, folder)
        count = count_videos(folder_path)
        status = "✓" if count >= num_videos else "✗"
        print(f"  {status} {folder}: {count} videos (need {num_videos})")
        if count < num_videos:
            sufficient = False
    
    for folder in EXPECTED_FOLDERS['real']:
        folder_path = os.path.join(DATASET_PATH, folder)
        count = count_videos(folder_path)
        status = "✓" if count >= num_videos else "✗"
        print(f"  {status} {folder}: {count} videos (need {num_videos})")
        if count < num_videos:
            sufficient = False
    
    if sufficient:
        print(f"\n✓ Dataset has enough videos for quick test with {num_videos} videos per class")
        print(f"  Quick test will use ~{num_videos * 7} videos total")
        return True
    else:
        print(f"\n⚠️  Some folders don't have {num_videos} videos")
        print(f"  Consider using fewer videos for quick test")
        return False


if __name__ == "__main__":
    print("\n" + "🔍" * 30)
    print("FaceForensics++ Dataset Validator")
    print("🔍" * 30)
    
    # Run validation
    is_valid = validate_dataset()
    
    # Check quick test feasibility
    if is_valid:
        check_quick_test_sample(num_videos=10)
    
    print("\n" + "=" * 60 + "\n")
    
    # Exit code
    sys.exit(0 if is_valid else 1)
