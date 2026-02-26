import cv2
import numpy as np
import tensorflow as tf
from model_factory import build_swin_large

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)

def preprocess_frames(frames):
    # Resize frames to the input size expected by the model
    resized_frames = [cv2.resize(frame, (224, 224)) for frame in frames]  # Adjust size as needed
    return np.array(resized_frames)

def predict_video_content(model, video_path):
    frames = load_video(video_path)
    processed_frames = preprocess_frames(frames)
    predictions = model.predict(processed_frames)
    return predictions

def main(video_path, model_path):
    model = build_swin_large(input_shape=(224, 224, 3), num_classes=2)  # Adjust input shape and classes as needed
    model.load_weights(model_path)
    
    predictions = predict_video_content(model, video_path)
    for i, prediction in enumerate(predictions):
        label = 'Fake' if np.argmax(prediction) == 0 else 'Real'
        print(f'Frame {i}: {label}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test video content using the trained model.')
    parser.add_argument('video_path', type=str, help='Path to the input video file.')
    parser.add_argument('model_path', type=str, help='Path to the trained model weights.')
    args = parser.parse_args()
    
    main(args.video_path, args.model_path)