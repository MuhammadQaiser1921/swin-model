"""
Swin Transformer - Tiny for Deepfake Detection
===============================================

Uses keras_cv.SwinTransformer (primary) or timm (fallback).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_swin_tiny(input_shape, num_classes=2):
    """
    Build Swin-Tiny model using keras_cv.SwinTransformer.
    
    Args:
        input_shape: Tuple (height, width, channels) - e.g., (224, 224, 3)
        num_classes: Number of output classes (default: 2)
    
    Returns:
        Keras Model ready for training/inference
    """
    try:
        import keras_cv
        print("✓ Using keras_cv.models.SwinTransformer (Tiny)")
        
        backbone = keras_cv.models.SwinTransformer(
            include_top=False,
            input_shape=input_shape,
            architecture="tiny"
        )
        
        inputs = layers.Input(shape=input_shape)
        x = backbone(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name="SwinTiny")
        return model
        
    except ImportError:
        print("⚠️ keras_cv not available, trying timm...")
        try:
            import timm
            print("✓ Using timm.models.swin_tiny_patch4_window7_224")
            # Load timm model - requires pytorch backend
            model_timm = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)
            print("✓ Model loaded successfully")
            return model_timm
        except ImportError:
            raise ImportError("Neither keras_cv nor timm available. Install: pip install keras-cv timm")

