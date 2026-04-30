"""Swin Transformer building blocks and configurable model builders."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def window_partition(x, window_size):
    b = tf.shape(x)[0]
    h, w, c = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    ws = window_size
    x = tf.reshape(x, [b, h // ws, ws, w // ws, ws, c])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, [-1, ws, ws, c])


def window_unpartition(windows, window_size, h, w):
    b = tf.shape(windows)[0] // (h // window_size) // (w // window_size)
    ws = window_size
    x = tf.reshape(windows, [b, h // ws, w // ws, ws, ws, -1])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, [b, h, w, -1])


class LocalRefine(layers.Layer):
    def __init__(self, dim, activation="mish", **kwargs):
        super().__init__(**kwargs)
        self.dw = layers.DepthwiseConv2D(kernel_size=3, padding="same")
        self.pw = layers.Conv2D(dim, kernel_size=1, padding="same")
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.act = layers.Activation(activation)

    def call(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))


class SqueezeExcite(layers.Layer):
    def __init__(self, dim, ratio=8, activation="mish", **kwargs):
        super().__init__(**kwargs)
        hidden = max(dim // ratio, 32)
        self.pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.fc1 = layers.Conv2D(hidden, kernel_size=1, activation=activation)
        self.fc2 = layers.Conv2D(dim, kernel_size=1, activation="sigmoid")

    def call(self, x):
        w = self.pool(x)
        w = self.fc1(w)
        w = self.fc2(w)
        return x * w


class WindowAttention(layers.Layer):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        attention_activation="softmax",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.attention_activation = attention_activation

        self.relative_position_bias_table = self.add_weight(
            name="rel_pos_bias_table",
            shape=((2 * window_size - 1) * (2 * window_size - 1), num_heads),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )

        coords_h = np.arange(window_size)
        coords_w = np.arange(window_size)
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.transpose(1, 2, 0).reshape(-1, 2)
        relative_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1).astype(np.int32)
        self.rel_pos_index = tf.constant(relative_position_index, dtype=tf.int32)

        self.qkv = layers.Dense(dim * 3, use_bias=True)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)

    def _apply_attention_activation(self, attn):
        if self.attention_activation == "softmax":
            return tf.nn.softmax(attn, axis=-1)
        if self.attention_activation == "sigmoid":
            attn = tf.nn.sigmoid(attn)
            denom = tf.reduce_sum(attn, axis=-1, keepdims=True) + 1e-8
            return attn / denom
        raise ValueError(
            f"Unsupported attention_activation='{self.attention_activation}'. "
            "Use 'softmax' or 'sigmoid'."
        )

    def call(self, x, mask=None, training=False):
        b, n, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = tf.transpose(
            tf.reshape(self.qkv(x), [b, n, 3, self.num_heads, -1]), [2, 0, 3, 1, 4]
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ tf.transpose(k, [0, 1, 3, 2])

        rel_pos_bias = tf.gather(
            self.relative_position_bias_table, tf.reshape(self.rel_pos_index, [-1])
        )
        rel_pos_bias = tf.reshape(rel_pos_bias, [n, n, self.num_heads])
        attn = attn + tf.expand_dims(tf.transpose(rel_pos_bias, [2, 0, 1]), 0)

        if mask is not None:
            attn = attn + mask

        attn = self._apply_attention_activation(attn)
        attn = self.attn_drop(attn, training=training)

        x = tf.reshape(tf.transpose(attn @ v, [0, 2, 1, 3]), [b, n, c])
        x = self.proj(x)
        return self.proj_drop(x, training=training)


class SwinTransformerBlock(layers.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        mlp_activation="mish",
        attention_activation="softmax",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size,
            num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            attention_activation=attention_activation,
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = keras.Sequential(
            [
                layers.Dense(int(dim * mlp_ratio), activation=mlp_activation),
                layers.Dropout(drop),
                layers.Dense(dim),
                layers.Dropout(drop),
            ]
        )

    def call(self, x, training=False):
        h, w, c = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        shortcut = x
        x = self.norm1(x)
        x_win = window_partition(x, self.window_size)
        x_win = tf.reshape(x_win, [-1, self.window_size * self.window_size, c])
        x_win = self.attn(x_win, training=training)
        x_win = tf.reshape(x_win, [-1, self.window_size, self.window_size, c])
        x = window_unpartition(x_win, self.window_size, h, w)
        x = shortcut + x
        return x + self.mlp(self.norm2(x), training=training)


class PatchMerging(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.reduction = layers.Dense(2 * dim, use_bias=False)
        self.norm = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x):
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        return self.reduction(self.norm(x))


def build_swin_tiny(
    input_shape,
    num_classes=2,
    mlp_activation="mish",
    attention_activation="softmax",
    output_activation="sigmoid",
    block_dropout=0.1,
    classifier_dropout=0.2,
    classifier_hidden_dims=(512, 128),
    use_local_refine=True,
    use_se=True,
    se_ratio=8,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    if input_shape[-1] == 1:
        x = layers.Conv2D(3, kernel_size=1)(x)

    x = layers.Conv2D(96, kernel_size=4, strides=4)(x)
    x = layers.LayerNormalization()(x)

    dims = [96, 192, 384, 768]
    heads = [3, 6, 12, 24]
    for stage_idx in range(4):
        for _ in range(2):
            x = SwinTransformerBlock(
                dim=dims[stage_idx],
                num_heads=heads[stage_idx],
                window_size=7,
                mlp_activation=mlp_activation,
                attention_activation=attention_activation,
                drop=block_dropout,
                attn_drop=block_dropout,
            )(x)

        if use_local_refine:
            x = x + LocalRefine(dims[stage_idx], activation=mlp_activation)(x)
        if use_se:
            x = SqueezeExcite(dims[stage_idx], ratio=se_ratio, activation=mlp_activation)(x)

        if stage_idx < 3:
            x = PatchMerging(dim=dims[stage_idx])(x)

    x = layers.GlobalAveragePooling2D()(x)
    for hidden_dim in classifier_hidden_dims:
        x = layers.Dense(hidden_dim, activation=mlp_activation)(x)
        x = layers.Dropout(classifier_dropout)(x)

    outputs = layers.Dense(num_classes, activation=output_activation)(x)
    return keras.Model(inputs=inputs, outputs=outputs)


def build_audio_swin(
    input_shape=(224, 224, 3),
    num_classes=1,
    mlp_activation="mish",
    attention_activation="softmax",
    output_activation="sigmoid",
    classifier_hidden_dims=(512, 128),
    classifier_dropout=0.2,
):
    return build_swin_tiny(
        input_shape=input_shape,
        num_classes=num_classes,
        mlp_activation=mlp_activation,
        attention_activation=attention_activation,
        output_activation=output_activation,
        classifier_hidden_dims=classifier_hidden_dims,
        classifier_dropout=classifier_dropout,
        use_local_refine=True,
        use_se=True,
    )


def build_video_swin(
    input_shape=(224, 224, 3),
    num_classes=1,
    mlp_activation="mish",
    attention_activation="softmax",
    output_activation="sigmoid",
    classifier_hidden_dims=(512, 128),
    classifier_dropout=0.2,
):
    return build_swin_tiny(
        input_shape=input_shape,
        num_classes=num_classes,
        mlp_activation=mlp_activation,
        attention_activation=attention_activation,
        output_activation=output_activation,
        classifier_hidden_dims=classifier_hidden_dims,
        classifier_dropout=classifier_dropout,
        use_local_refine=True,
        use_se=True,
    )