"""
Swin Transformer implementation for TensorFlow/Keras
Designed to be compatible with both Video and Audio inputs.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def window_partition(x, window_size):
    """Partition x into windows: (B, H, W, C) -> (num_windows*B, ws, ws, C)"""
    B = tf.shape(x)[0]
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]
    C = tf.shape(x)[3]
    ws = window_size
    
    x = tf.reshape(x, [B, H // ws, ws, W // ws, ws, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [-1, ws, ws, C])
    return x


def window_unpartition(windows, window_size, H, W):
    """Reverse partition: (num_windows*B, ws, ws, C) -> (B, H, W, C)"""
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
        
        # Precompute relative position index using numpy
        coords_h = np.arange(window_size)
        coords_w = np.arange(window_size)
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij')) 
        coords_flatten = coords.transpose(1, 2, 0).reshape(-1, 2)
        relative_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1).astype(np.int32)
        
        # Register index as a non-trainable weight to ensure proper GPU placement
        self.relative_position_index = self.add_weight(
            name='relative_position_index',
            shape=relative_position_index.shape,
            initializer=keras.initializers.Constant(relative_position_index),
            trainable=False,
            dtype='int32'
        )

        self.qkv = layers.Dense(dim * 3, use_bias=True)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)
    
    def call(self, x, mask=None, training=False):
        B_ = tf.shape(x)[0]
        N = tf.shape(x)[1]
        C = tf.shape(x)[2]
        
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B_, N, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4]) 
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ tf.transpose(k, [0, 1, 3, 2])) * self.scale
        
        # FIX: Force the index to be treated as a tensor during call to ensure GPU placement
        rel_pos_index = tf.cast(self.relative_position_index, dtype=tf.int32)
        
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            tf.reshape(rel_pos_index, [-1])
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
        
        x = tf.reshape(tf.transpose(attn @ v, [0, 2, 1, 3]), [B_, N, C])
        x = self.proj_drop(self.proj(x), training=training)
        return x


class SwinTransformerBlock(layers.Layer):
    """Swin Transformer Block with dynamic resolution handling"""
    
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
        self.mlp = keras.Sequential([
            layers.Dense(int(dim * mlp_ratio), activation='gelu'),
            layers.Dropout(drop),
            layers.Dense(dim),
            layers.Dropout(drop)
        ])
        
        self.drop_path = layers.Dropout(drop_path) if drop_path > 0. else layers.Identity()
    
    def call(self, x, training=False):
        H, W = tf.shape(x)[1], tf.shape(x)[2]
        C = tf.shape(x)[3]
        
        x_norm = self.norm1(x)
        x_partitioned = window_partition(x_norm, self.window_size)
        x_partitioned = tf.reshape(x_partitioned, [-1, self.window_size*self.window_size, C])
        
        x_attn = self.attn(x_partitioned, training=training)
        x_attn = tf.reshape(x_attn, [-1, self.window_size, self.window_size, C])
        x_attn = window_unpartition(x_attn, self.window_size, H, W)
        
        x = x + self.drop_path(x_attn, training=training)
        x = x + self.drop_path(self.mlp(self.norm2(x), training=training), training=training)
        return x


class PatchMerging(layers.Layer):
    """Dynamic resolution reduction and channel expansion layer"""
    
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.reduction = layers.Dense(2 * dim, use_bias=False)
        self.norm = layers.LayerNormalization(epsilon=1e-5)
    
    def call(self, x):
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = self.norm(x)
        return self.reduction(x)


class PatchEmbedding(layers.Layer):
    """Initial patch embedding layer"""
    
    def __init__(self, patch_size=4, embed_dim=96, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = layers.Conv2D(
            embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid'
        )
        self.norm = layers.LayerNormalization(epsilon=1e-5)
    
    def call(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


def build_swin_tiny(input_shape, num_classes=2):
    """
    Constructs the Swin-Tiny architecture.
    Works for both Video (224, 224, 3) and Audio (128, 128, 1) inputs.
    """
    print(f"✓ Building Swin-Tiny Transformer for shape: {input_shape}")
    
    embed_dim = 96
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    window_size = 7
    
    inputs = keras.Input(shape=input_shape)
    
    # Optional: If audio input is single-channel, expand to 3 for standard processing
    x = inputs
    if input_shape[-1] == 1:
        x = layers.Conv2D(3, kernel_size=1)(x)

    x = PatchEmbedding(patch_size=4, embed_dim=embed_dim)(x)
    
    current_dim = embed_dim
    for stage_idx, (depth, num_head) in enumerate(zip(depths, num_heads)):
        for block_idx in range(depth):
            x = SwinTransformerBlock(
                dim=current_dim,
                num_heads=num_head,
                window_size=window_size,
                drop=0.1,
                drop_path=0.1,
                name=f"swin_block_s{stage_idx}_b{block_idx}"
            )(x)
        
        if stage_idx < len(depths) - 1:
            x = PatchMerging(dim=current_dim, name=f"patch_merge_{stage_idx}")(x)
            current_dim = current_dim * 2
    
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name='SwinTiny')