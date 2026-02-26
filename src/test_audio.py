import numpy as np
import cv2
import os
from model_factory import build_swin_large
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_spectrogram(file_path):
    # Load the spectrogram image
    img = load_img(file_path, target_size=(224, 224))  # Adjust size as needed
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    return img_array

def predict_audio(model, spectrogram_path):
    spectrogram = load_spectrogram(spectrogram_path)
    prediction = model.predict(spectrogram)
    return prediction

def main(audio_dir, model_path):
    # Load the trained model
    model = build_swin_large(input_shape=(224, 224, 3), num_classes=2)  # Adjust as needed
    model.load_weights(model_path)

    # Iterate through audio spectrogram files
    for filename in os.listdir(audio_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):  # Adjust extensions as needed
            file_path = os.path.join(audio_dir, filename)
            prediction = predict_audio(model, file_path)
            label = 'fake' if np.argmax(prediction) == 0 else 'real'
            print(f'File: {filename}, Prediction: {label}')

if __name__ == '__main__':
    audio_directory = 'data/audio'  # Adjust path as needed
    model_weights_path = 'models/checkpoints/swin_large_audio_model.h5'  # Adjust path as needed
    main(audio_directory, model_weights_path)