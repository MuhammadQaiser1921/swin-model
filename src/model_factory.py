"""
Model factory - builds Swin-Tiny using keras_cv
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_cv


def build_swin_tiny(input_shape, num_classes=2):
    """
    Build Swin-Tiny model using keras_cv.SwinTransformer.
    
    Args:
        input_shape: Tuple (height, width, channels)
        num_classes: Number of output classes (default: 2)
    
    Returns:
        Keras Model
    """
    print("✓ Building Swin-Tiny with keras_cv")
    
    # Backbone
    backbone = keras_cv.models.SwinTransformer(
        include_top=False,
        input_shape=input_shape,
        architecture="tiny"
    )
    
    # Classification head
    inputs = layers.Input(shape=input_shape)
    x = backbone(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="SwinTiny")
    return model

