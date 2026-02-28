import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def window_partition(x, window_size):
    """Partition x into windows: (B, H, W, C) -> (num_windows*B, ws, ws, C)"""
    B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    x = tf.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, [-1, window_size, window_size, C])

def window_unpartition(windows, window_size, H, W):
    """Reverse partition: (num_windows*B, ws, ws, C) -> (B, H, W, C)"""
    B = tf.shape(windows)[0] // (H // window_size) // (W // window_size)
    x = tf.reshape(windows, [B, H // window_size, W // window_size, window_size, window_size, -1])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, [B, H, W, -1])

class WindowAttention(layers.Layer):
    """Multi-head self-attention with relative position bias"""
    def __init__(self, dim, window_size, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_bias_table = self.add_weight(
            name='rel_pos_bias_table',
            shape=((2 * window_size - 1) * (2 * window_size - 1), num_heads),
            initializer='truncated_normal',
            trainable=True
        )

        # Pure TensorFlow index calculation to ensure GPU placement
        h, w = tf.range(window_size), tf.range(window_size)
        grid_h, grid_w = tf.meshgrid(h, w, indexing='ij')
        coords = tf.stack([grid_h, grid_w], axis=-1)
        coords_flatten = tf.reshape(coords, [-1, 2])
        relative_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]
        relative_coords += window_size - 1
        
        rel_h = relative_coords[:, :, 0] * (2 * window_size - 1)
        rel_w = relative_coords[:, :, 1]
        self.rel_pos_index = tf.cast(rel_h + rel_w, tf.int32)

        self.qkv = layers.Dense(dim * 3, use_bias=True)
        self.proj = layers.Dense(dim)

    def call(self, x):
        B_, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = tf.transpose(tf.reshape(self.qkv(x), [B_, N, 3, self.num_heads, -1]), [2, 0, 3, 1, 4])
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ tf.transpose(k, [0, 1, 3, 2]))
        
        # Gathering relative bias using the GPU-safe index
        bias = tf.gather(self.relative_position_bias_table, tf.reshape(self.rel_pos_index, [-1]))
        bias = tf.reshape(bias, [N, N, self.num_heads])
        attn = attn + tf.expand_dims(tf.transpose(bias, [2, 0, 1]), 0)

        attn = tf.nn.softmax(attn, axis=-1)
        x = tf.reshape(tf.transpose(attn @ v, [0, 2, 1, 3]), [B_, N, C])
        return self.proj(x)

class SwinBlock(layers.Layer):
    """Consolidated Swin Transformer Block"""
    def __init__(self, dim, num_heads, window_size=7, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = keras.Sequential([
            layers.Dense(4 * dim, activation='gelu'),
            layers.Dense(dim)
        ])

    def call(self, x):
        H, W, C = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        shortcut = x
        x = self.norm1(x)
        x_win = window_partition(x, 7)
        x_win = self.attn(tf.reshape(x_win, [-1, 49, tf.shape(x_win)[-1]]))
        x_win = tf.reshape(x_win, [-1, 7, 7, tf.shape(x_win)[-1]])
        x = window_unpartition(x_win, 7, H, W)
        x = shortcut + x
        return x + self.mlp(self.norm2(x))

class PatchMerging(layers.Layer):
    """Downsamples resolution by 2x and doubles channel depth"""
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.reduction = layers.Dense(2 * dim, use_bias=False)
        self.norm = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x):
        x0, x1, x2, x3 = x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :], x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        return self.reduction(self.norm(x))

def build_model(input_shape, num_classes):
    """
    Returns a Swin-Tiny model.
    Compatible with Video (224, 224, 3) or Audio (128, 128, 1).
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Expand channels if input is single-channel audio
    if input_shape[-1] == 1:
        x = layers.Conv2D(3, kernel_size=1)(x)
        
    x = layers.Conv2D(96, kernel_size=4, strides=4)(x)
    x = layers.LayerNormalization()(x)
    
    dims, heads = [96, 192, 384, 768], [3, 6, 12, 24]
    
    for i in range(4):
        for _ in range(2):
            x = SwinBlock(dim=dims[i], num_heads=heads[i])(x)
        if i < 3:
            x = PatchMerging(dims[i])(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name="SwinTiny_Custom")
    return model