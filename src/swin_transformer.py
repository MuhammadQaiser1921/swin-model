"""
Swin Transformer implementation for TensorFlow/Keras
Fixed for GPU/CPU Device Placement Errors.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def window_partition(x, window_size):
    B = tf.shape(x)[0]
    H, W, C = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    ws = window_size
    x = tf.reshape(x, [B, H // ws, ws, W // ws, ws, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, [-1, ws, ws, C])

def window_unpartition(windows, window_size, H, W):
    B = tf.shape(windows)[0] // (H // window_size) // (W // window_size)
    ws = window_size
    x = tf.reshape(windows, [B, H // ws, W // ws, ws, ws, -1])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, [B, H, W, -1])

class WindowAttention(layers.Layer):
    def __init__(self, dim, window_size, num_heads, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_bias_table = self.add_weight(
            name='rel_pos_bias_table',
            shape=((2 * window_size - 1) * (2 * window_size - 1), num_heads),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )
        
        # Precompute index as a constant to avoid CPU/GPU access errors
        coords_h = np.arange(window_size)
        coords_w = np.arange(window_size)
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij')) 
        coords_flatten = coords.transpose(1, 2, 0).reshape(-1, 2)
        relative_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1).astype(np.int32)
        
        # CRITICAL FIX: Use tf.constant instead of self.add_weight for the index
        self.rel_pos_index = tf.constant(relative_position_index, dtype=tf.int32)

        self.qkv = layers.Dense(dim * 3, use_bias=True)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)

    def call(self, x, mask=None, training=False):
        B_, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = tf.transpose(tf.reshape(self.qkv(x), [B_, N, 3, self.num_heads, -1]), [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ tf.transpose(k, [0, 1, 3, 2]))
        
        # Access the constant index safely
        rel_pos_bias = tf.gather(self.relative_position_bias_table, tf.reshape(self.rel_pos_index, [-1]))
        rel_pos_bias = tf.reshape(rel_pos_bias, [N, N, self.num_heads])
        attn = attn + tf.expand_dims(tf.transpose(rel_pos_bias, [2, 0, 1]), 0)
        
        if mask is not None: attn = attn + mask
        
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)
        
        x = tf.reshape(tf.transpose(attn @ v, [0, 2, 1, 3]), [B_, N, C])
        return self.proj_drop(self.proj(x), training=training)

class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(dim, window_size, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = keras.Sequential([
            layers.Dense(int(dim * mlp_ratio), activation='gelu'),
            layers.Dropout(drop),
            layers.Dense(dim),
            layers.Dropout(drop)
        ])
        self.drop_path = layers.Identity() # Simplified for stability

    def call(self, x, training=False):
        H, W, C = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        shortcut = x
        x = self.norm1(x)
        x_win = window_partition(x, self.window_size)
        x_win = tf.reshape(x_win, [-1, self.window_size*self.window_size, C])
        x_win = self.attn(x_win, training=training)
        x_win = tf.reshape(x_win, [-1, self.window_size, self.window_size, C])
        x = window_unpartition(x_win, self.window_size, H, W)
        x = shortcut + x
        return x + self.mlp(self.norm2(x), training=training)

class PatchMerging(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.reduction = layers.Dense(2 * dim, use_bias=False)
        self.norm = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x):
        x0, x1, x2, x3 = x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :], x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        return self.reduction(self.norm(x))

def build_swin_tiny(input_shape, num_classes=2):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    if input_shape[-1] == 1: # Audio channel expansion
        x = layers.Conv2D(3, kernel_size=1)(x)

    x = layers.Conv2D(96, kernel_size=4, strides=4)(x)
    x = layers.LayerNormalization()(x)
    
    dims, heads = [96, 192, 384, 768], [3, 6, 12, 24]
    for i in range(4):
        for j in range(2):
            x = SwinTransformerBlock(dim=dims[i], num_heads=heads[i], window_size=7)(x)
        if i < 3: x = PatchMerging(dim=dims[i])(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    return keras.Model(inputs=inputs, outputs=layers.Dense(num_classes, activation='softmax')(x))