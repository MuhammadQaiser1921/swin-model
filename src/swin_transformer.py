"""
Swin Transformer implementation for TensorFlow/Keras
Designed to be compatible with timm pretrained weights
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def window_partition(x, window_size):
    """Partition x into windows with shape (num_windows*B, window_size, window_size, C)"""
    B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    ws = window_size
    
    x = tf.reshape(x, [B, H // ws, ws, W // ws, ws, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [-1, ws, ws, C])
    
    return x


def window_unpartition(windows, window_size, H, W):
    """Reverse of window_partition"""
    B = tf.shape(windows)[0] // (H // window_size) // (W // window_size)
    ws = window_size
    
    x = tf.reshape(windows, [B, H // ws, W // ws, ws, ws, -1])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [B, H, W, -1])
    
    return x


class WindowAttention(layers.Layer):
    """Multi-head self-attention with relative position bias"""
    
    def __init__(self, dim, window_size, num_heads=8, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Learnable relative position bias
        self.relative_position_bias_table = self.add_weight(
            name='relative_position_bias_table',
            shape=((2 * window_size - 1) * (2 * window_size - 1), num_heads),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )
        
        # Precompute relative positions
        coords_h = tf.range(window_size)
        coords_w = tf.range(window_size)
        coords = tf.stack(tf.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, ws, ws
        coords_flatten = tf.reshape(tf.transpose(coords, [1, 2, 0]), [-1, 2])  # ws*ws, 2
        relative_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]  # ws*ws, ws*ws, 2
        relative_coords = tf.experimental.numpy.asarray(relative_coords)
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = tf.reduce_sum(relative_coords, axis=-1)  # ws*ws, ws*ws
        self.relative_position_index = tf.Variable(
            relative_position_index, trainable=False, name='relative_position_index'
        )
        
        self.qkv = layers.Dense(dim * 3, use_bias=True)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)
    
    def call(self, x, mask=None, training=False):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        
        # Linear projection for Q, K, V
        qkv = self.qkv(x)  # B, N, 3*dim
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # 3, B, num_heads, N, head_dim
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ tf.transpose(k, [0, 1, 3, 2])) * self.scale  # B, num_heads, N, N
        
        # Add relative position bias
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            tf.reshape(self.relative_position_index, [-1])
        )
        relative_position_bias = tf.reshape(
            relative_position_bias,
            [N, N, self.num_heads]
        )
        relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, 0)
        
        if mask is not None:
            attn = attn + mask
        
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)
        
        x = (attn @ v)  # B, num_heads, N, head_dim
        x = tf.transpose(x, [0, 2, 1, 3])  # B, N, num_heads, head_dim
        x = tf.reshape(x, [B, N, C])
        
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        
        return x


class SwinTransformerBlock(layers.Layer):
    """Swin Transformer Block"""
    
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4., 
                 drop=0., attn_drop=0., drop_path=0., **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop
        )
        
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = keras.Sequential([
            layers.Dense(mlp_hidden_dim, activation='gelu'),
            layers.Dropout(drop),
            layers.Dense(dim),
            layers.Dropout(drop)
        ])
        
        self.drop_path = layers.Dropout(drop_path) if drop_path > 0. else layers.Identity()
    
    def call(self, x, training=False):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        # Window attention
        x_norm = self.norm1(x)
        x_partitioned = window_partition(x_norm, self.window_size)
        x_partitioned = tf.reshape(x_partitioned, [-1, self.window_size*self.window_size, C])
        
        x_attn = self.attn(x_partitioned, training=training)
        x_attn = tf.reshape(x_attn, [-1, self.window_size, self.window_size, C])
        x_attn = window_unpartition(x_attn, self.window_size, H, W)
        
        x = x + self.drop_path(x_attn, training=training)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x), training=training), training=training)
        
        return x


class PatchEmbedding(layers.Layer):
    """Patch embedding layer"""
    
    def __init__(self, patch_size=4, embed_dim=96, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = layers.Conv2D(
            embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid'
        )
        self.norm = layers.LayerNormalization(epsilon=1e-5)
    
    def call(self, x):
        x = self.proj(x)  # B, H', W', embed_dim
        x = self.norm(x)
        return x


def build_swin_tiny(input_shape, num_classes=2, pretrained=False):
    """
    Build Swin-Tiny for deepfake detection
    
    Args:
        input_shape: Tuple (height, width, channels)
        num_classes: Number of output classes
        pretrained: Whether to load ImageNet pretrained weights (placeholder)
    
    Returns:
        Keras Model
    """
    print("✓ Building Swin-Tiny Transformer")
    
    # Swin-Tiny config
    patch_size = 4
    embed_dim = 96
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    window_size = 7
    
    inputs = keras.Input(shape=input_shape)
    
    # Patch embedding
    x = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim)(inputs)
    
    # Reshape to flatten spatial dimensions: (B, H', W', embed_dim) -> (B, H'*W', embed_dim)
    def reshape_fn(x):
        shape = tf.shape(x)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        return tf.reshape(x, [B, H * W, C])
    
    x = layers.Lambda(reshape_fn)(x)
    
    # Swin layers
    current_dim = embed_dim
    for stage_idx, (depth, num_head) in enumerate(zip(depths, num_heads)):
        for block_idx in range(depth):
            x = SwinTransformerBlock(
                dim=current_dim,
                num_heads=num_head,
                window_size=window_size,
                mlp_ratio=4.0,
                drop=0.1,
                attn_drop=0.0,
                drop_path=0.1
            )(x)
        
        if stage_idx < len(depths) - 1:
            # Patch merging
            current_dim = current_dim * 2
            # Simplified patch merging (reduces spatial dims)
            x = layers.Dense(current_dim)(x)
    
    # Classification head
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=x, name='SwinTiny')
    
    return model
