"""
Swin Transformer implementation 
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# =========================
# ACTIVATION FUNCTIONS
# =========================
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


def get_activation(name):
    if name == "gelu":
        return tf.nn.gelu
    elif name == "relu":
        return tf.nn.relu
    elif name == "swish":
        return tf.nn.swish
    elif name == "mish":
        return mish
    else:
        return tf.nn.gelu


# =========================
# WINDOW PARTITION
# =========================
def window_partition(x, window_size):
    B = tf.shape(x)[0]
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]
    C = tf.shape(x)[3]

    x = tf.reshape(x, [B, H // window_size, window_size,
                       W // window_size, window_size, C])

    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])

    windows = tf.reshape(x, [-1, window_size, window_size, C])

    return windows


def window_unpartition(windows, window_size, H, W):

    B = tf.shape(windows)[0] // ((H // window_size) * (W // window_size))

    x = tf.reshape(
        windows,
        [B, H // window_size, W // window_size,
         window_size, window_size, -1]
    )

    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])

    x = tf.reshape(x, [B, H, W, -1])

    return x


# =========================
# WINDOW ATTENTION
# =========================
class WindowAttention(layers.Layer):

    def __init__(self, dim, window_size, num_heads, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.scale = (dim // num_heads) ** -0.5

        self.qkv = layers.Dense(dim * 3)
        self.proj = layers.Dense(dim)

    def call(self, x):

        B = tf.shape(x)[0]
        N = tf.shape(x)[1]
        C = tf.shape(x)[2]

        qkv = self.qkv(x)

        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, C // self.num_heads])

        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])

        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = tf.matmul(q, k, transpose_b=True)

        attn = tf.nn.softmax(attn, axis=-1)

        x = tf.matmul(attn, v)

        x = tf.transpose(x, [0, 2, 1, 3])

        x = tf.reshape(x, [B, N, C])

        x = self.proj(x)

        return x


# =========================
# SWIN TRANSFORMER BLOCK
# =========================
class SwinTransformerBlock(layers.Layer):

    def __init__(
            self,
            dim,
            num_heads,
            window_size=7,
            mlp_ratio=4,
            activation="gelu",
            **kwargs):

        super().__init__(**kwargs)

        self.dim = dim
        self.window_size = window_size

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)

        self.attn = WindowAttention(dim, window_size, num_heads)

        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        act = get_activation(activation)

        self.mlp = keras.Sequential([

            layers.Dense(int(dim * mlp_ratio), activation=act),

            layers.Dropout(0.1),

            layers.Dense(dim),

            layers.Dropout(0.1)

        ])

    def call(self, x):

        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = tf.shape(x)[3]

        shortcut = x

        x = self.norm1(x)

        x_windows = window_partition(x, self.window_size)

        x_windows = tf.reshape(
            x_windows,
            [-1, self.window_size * self.window_size, C]
        )

        attn_windows = self.attn(x_windows)

        attn_windows = tf.reshape(
            attn_windows,
            [-1, self.window_size, self.window_size, C]
        )

        x = window_unpartition(attn_windows, self.window_size, H, W)

        x = shortcut + x

        x = x + self.mlp(self.norm2(x))

        return x


# =========================
# PATCH MERGING
# =========================
class PatchMerging(layers.Layer):

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)

        self.norm = layers.LayerNormalization(epsilon=1e-5)

        self.reduction = layers.Dense(2 * dim, use_bias=False)

    def call(self, x):

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = tf.concat([x0, x1, x2, x3], axis=-1)

        x = self.norm(x)

        x = self.reduction(x)

        return x


# =========================
# BUILD SWIN TINY
# =========================
def build_swin_tiny(
        input_shape=(224, 224, 3),
        num_classes=1,
        activation="gelu"):

    inputs = keras.Input(shape=input_shape)

    x = inputs

    if input_shape[-1] == 1:
        x = layers.Conv2D(3, 1)(x)

    # Patch Embedding
    x = layers.Conv2D(96, kernel_size=4, strides=4)(x)

    x = layers.LayerNormalization()(x)

    dims = [96, 192, 384, 768]
    heads = [3, 6, 12, 24]

    for i in range(4):

        for _ in range(2):

            x = SwinTransformerBlock(
                dim=dims[i],
                num_heads=heads[i],
                window_size=7,
                activation=activation
            )(x)

        if i < 3:
            x = PatchMerging(dims[i])(x)

    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)

    return model
