"""
Model Loading & Inference Script
=================================

Load trained Swin-Tiny model weights and perform inference on new images/videos.

Usage:
    python load_model.py --weights path/to/weights.h5 --image path/to/image.jpg
    python load_model.py --model path/to/savedmodel/ --video path/to/video.mp4
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import json
from pathlib import Path


def load_model_from_weights(weights_path, model_factory_path='model_factory.py'):
    """
    Load model architecture and weights.
    
    Args:
        weights_path: Path to .h5 weights file
        model_factory_path: Path to model_factory.py
    
    Returns:
        Loaded Keras model
    """
    import sys
    sys.path.insert(0, os.path.dirname(model_factory_path))
    
    from model_factory import build_swin_tiny
    
    print(f"📦 Loading model from weights: {weights_path}")
    
    # Build architecture
    model = build_swin_tiny(input_shape=(224, 224, 3), num_classes=2)
    
    # Load weights
    model.load_weights(weights_path)
    
    print("✓ Model loaded successfully")
    return model


def load_model_savedmodel(model_path):
    """
    Load model from SavedModel format.
    
    Args:
        model_path: Path to SavedModel directory
    
    Returns:
        Loaded Keras model
    """
    print(f"📦 Loading model from SavedModel: {model_path}")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully")
    return model


def load_model_h5(model_path):
    """
    Load complete model from .h5 file.
    
    Args:
        model_path: Path to .h5 model file
    
    Returns:
        Loaded Keras model
    """
    print(f"📦 Loading model from H5: {model_path}")
    model = keras.models.load_model(model_path, custom_objects={
        'GELU': keras.activations.gelu
    })
    print("✓ Model loaded successfully")
    return model


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess image for inference.
    
    Args:
        image_path: Path to image file
        target_size: Size to resize to
    
    Returns:
        Preprocessed image array (1, 224, 224, 3)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img


def extract_frames_from_video(video_path, num_frames=10, target_size=(224, 224)):
    """
    Extract frames from video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        target_size: Size to resize frames to
    
    Returns:
        Preprocessed frames array (num_frames, 224, 224, 3)
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError(f"Could not read video: {video_path}")
    
    # Sample frame indices uniformly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        frame = cv2.resize(frame, target_size)
        
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        
        frames.append(frame)
    
    cap.release()
    return np.array(frames)


def predict_image(model, image_path):
    """
    Predict class for single image.
    
    Args:
        model: Trained model
        image_path: Path to image
    
    Returns:
        Dictionary with prediction results
    """
    print(f"\n🖼️  Processing image: {image_path}")
    
    # Preprocess
    img = preprocess_image(image_path)
    
    # Predict
    pred = model.predict(img, verbose=0)
    pred_proba = pred[0]
    pred_class = np.argmax(pred_proba)
    pred_confidence = pred_proba[pred_class]
    
    # Interpret
    class_names = ['FAKE', 'REAL']
    prediction = class_names[pred_class]
    
    result = {
        'image': image_path,
        'prediction': prediction,
        'confidence': float(pred_confidence),
        'probabilities': {
            'fake': float(pred_proba[0]),
            'real': float(pred_proba[1])
        }
    }
    
    return result


def predict_video(model, video_path, num_frames=10, aggregate='mean'):
    """
    Predict class for video (aggregate frame predictions).
    
    Args:
        model: Trained model
        video_path: Path to video
        num_frames: Number of frames to extract
        aggregate: Aggregation method ('mean', 'max', 'voting')
    
    Returns:
        Dictionary with prediction results
    """
    print(f"\n🎥 Processing video: {video_path}")
    
    # Extract frames
    frames = extract_frames_from_video(video_path, num_frames)
    print(f"   Extracted {len(frames)} frames")
    
    # Predict for each frame
    frame_preds = model.predict(frames, verbose=0)
    frame_proba = frame_preds[:, 1]  # Probability of class 1 (REAL)
    
    # Aggregate predictions
    if aggregate == 'mean':
        final_proba = np.mean(frame_proba)
    elif aggregate == 'max':
        final_proba = np.max(frame_proba)
    elif aggregate == 'voting':
        # Majority voting
        frame_classes = (frame_proba >= 0.5).astype(int)
        final_proba = np.sum(frame_classes) / len(frame_classes)
    else:
        final_proba = np.mean(frame_proba)
    
    # Determine final class
    final_class = 1 if final_proba >= 0.5 else 0
    class_names = ['FAKE', 'REAL']
    
    result = {
        'video': video_path,
        'num_frames': len(frames),
        'frame_predictions': frame_proba.tolist(),
        'aggregation_method': aggregate,
        'final_prediction': class_names[final_class],
        'confidence': float(final_proba) if final_class == 1 else float(1 - final_proba),
        'probabilities': {
            'fake': float(1 - final_proba),
            'real': float(final_proba)
        }
    }
    
    return result


def print_results(results):
    """Pretty print prediction results"""
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    if isinstance(results, list):
        for r in results:
            print(f"\n📄 {r.get('image') or r.get('video')}:")
            print(f"   Prediction: {r['prediction']} ({r['final_prediction']})")
            print(f"   Confidence: {r['confidence']:.4f}")
            print(f"   Probabilities:")
            print(f"     - Fake: {r['probabilities'].get('fake', r['probabilities'][0]):.4f}")
            print(f"     - Real: {r['probabilities'].get('real', r['probabilities'][1]):.4f}")
    else:
        print(f"\n📄 {results.get('image') or results.get('video')}:")
        print(f"   Prediction: {results['prediction']}")
        print(f"   Confidence: {results['confidence']:.4f}")
        print(f"   Probabilities:")
        print(f"     - Fake: {results['probabilities']['fake']:.4f}")
        print(f"     - Real: {results['probabilities']['real']:.4f}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Video Model Inference Tool')
    
    # Model arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--weights', type=str, help='Path to .h5 weights file')
    model_group.add_argument('--savedmodel', type=str, help='Path to SavedModel directory')
    model_group.add_argument('--model', type=str, help='Path to .h5 model file')
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Path to image file')
    input_group.add_argument('--video', type=str, help='Path to video file')
    input_group.add_argument('--image-dir', type=str, help='Directory of images')
    input_group.add_argument('--video-dir', type=str, help='Directory of videos')
    
    # Optional arguments
    parser.add_argument('--model-factory', type=str, default='model_factory.py',
                       help='Path to model_factory.py (required with --weights)')
    parser.add_argument('--num-frames', type=int, default=10,
                       help='Number of frames to extract from video')
    parser.add_argument('--aggregate', type=str, default='mean',
                       choices=['mean', 'max', 'voting'],
                       help='Aggregation method for video predictions')
    parser.add_argument('--output', type=str, help='Path to save results as JSON')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Decision threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    # Load model
    if args.weights:
        model = load_model_from_weights(args.weights, args.model_factory)
    elif args.savedmodel:
        model = load_model_savedmodel(args.savedmodel)
    else:
        model = load_model_h5(args.model)
    
    # Compile for inference
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Perform inference
    results = []
    
    if args.image:
        result = predict_image(model, args.image)
        results.append(result)
        print_results(result)
    
    elif args.video:
        result = predict_video(model, args.video, num_frames=args.num_frames,
                              aggregate=args.aggregate)
        results.append(result)
        print_results(result)
    
    elif args.image_dir:
        image_files = list(Path(args.image_dir).glob('*.jpg')) + \
                     list(Path(args.image_dir).glob('*.png'))
        for img_path in image_files:
            try:
                result = predict_image(model, str(img_path))
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        print_results(results)
    
    elif args.video_dir:
        video_files = list(Path(args.video_dir).glob('*.mp4')) + \
                     list(Path(args.video_dir).glob('*.avi'))
        for vid_path in video_files:
            try:
                result = predict_video(model, str(vid_path), num_frames=args.num_frames,
                                      aggregate=args.aggregate)
                results.append(result)
            except Exception as e:
                print(f"Error processing {vid_path}: {e}")
        print_results(results)
    
    # Save results if requested
    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to: {args.output}")


if __name__ == '__main__':
    main()
