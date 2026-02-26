"""
Swin Transformer - Tiny Configuration for Deepfake Detection Research
=======================================================================

This module implements the Swin Transformer Tiny architecture, specifically
designed for multi-modal deepfake detection (audio & video) with XAI capabilities.

Architecture Overview:
- 4 hierarchical stages with progressive downsampling
- Window-based Multi-Head Self Attention (W-MSA)
- Shifted Window Attention (SW-MSA) for cross-window connections
- Relative position bias for translation equivariance
- Patch merging for dimensional expansion at each stage

Swin-Tiny is a lightweight variant (~28M parameters) ideal for:
- Fast training and inference
- Resource-constrained environments (mobile, edge devices)
- Quick experimentation and prototyping

References:
- Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
  (Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, et al., ICCV 2021)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# ========== SWIN-TINY CONFIGURATION ==========
# All hyperparameters follow the official Swin-Tiny specification
# Swin-Tiny: ~28M parameters (8x smaller than Swin-Large)

patch_size = (4, 4)              # Input patch size: 4x4 pixels per patch
dropout_rate = 0.1               # Stochastic dropout rate for regularization

# Hierarchical architecture parameters (Stage 1 → Stage 4)
num_heads_stages = [3, 6, 12, 24]        # Attention heads per stage (scales with depth)
embed_dim_stages = [96, 192, 384, 768]   # Embedding dimension per stage (doubles each stage)
num_mlp_stages = [384, 768, 1536, 3072]  # MLP hidden dimension (4x embedding dims)
depths = [2, 2, 6, 2]                    # Block count per stage (6 blocks at deepest stage, vs 18 for Large)

# Window attention parameters
window_size = 7                  # Local attention window size (7x7 patches)
shift_size = 3                   # Shift size for alternating SW-MSA (half of window_size)

# Attention mechanism parameters
qkv_bias = True                  # Include bias in Q, K, V projections
label_smoothing = 0.1            # Label smoothing for training stability

# ========== HELPER FUNCTIONS ==========

def window_partition(x, window_size):
    """
    Partition input tensor into non-overlapping square windows.
    
    This function reshapes the spatial dimensions into local windows for
    efficient window-based attention computation. Instead of attending to
    all tokens (O(n²) complexity), we attend only within windows (O(n)).
    
    Args:
        x: Input tensor of shape (batch, height, width, channels)
        window_size: Integer size of square windows (e.g., 7x7)
    
    Returns:
        Partitioned windows of shape (num_windows, window_size, window_size, channels)
        where num_windows = (batch * height // window_size * width // window_size)
    
    Example:
        Input: (1, 56, 56, 192) → Output: (49, 7, 7, 192)
    """
    _, height, width, channels = x.shape
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    # Reshape into [batch, height_patches, patch_size, width_patches, patch_size, channels]
    x = tf.reshape(
        x, shape=(-1, patch_num_y, window_size, patch_num_x, window_size, channels)
    )
    # Transpose to group by window: [batch, height_patches, width_patches, patch_size, patch_size, channels]
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    # Reshape into windows: [batch * height_patches * width_patches, patch_size, patch_size, channels]
    windows = tf.reshape(x, shape=(-1, window_size, window_size, channels))
    return windows


def window_reverse(windows, window_size, height, width, channels):
    """
    Reverse the window partitioning operation (reconstruct from windows).
    
    This is the inverse of window_partition(). Used to merge window outputs
    back into the spatial grid after attention computation.
    
    Args:
        windows: Partitioned windows of shape (num_windows, window_size, window_size, channels)
        window_size: Integer size of square windows
        height: Original spatial height
        width: Original spatial width
        channels: Number of channels
    
    Returns:
        Reconstructed tensor of shape (batch, height, width, channels)
    """
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    # Reshape back to [batch, height_patches, width_patches, patch_size, patch_size, channels]
    x = tf.reshape(
        windows,
        shape=(-1, patch_num_y, patch_num_x, window_size, window_size, channels),
    )
    # Transpose back: [batch, height_patches, patch_size, width_patches, patch_size, channels]
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    # Final reshape: [batch, height, width, channels]
    x = tf.reshape(x, shape=(-1, height, width, channels))
    return x


class DropPath(layers.Layer):
    """
    Stochastic Depth (DropPath) - randomly drop residual branches during training.
    
    This is a regularization technique that randomly drops entire residual paths
    with probability drop_prob. This encourages the network to learn more robust
    features and improves generalization. Unlike standard dropout, DropPath operates
    at the sample level, not element-wise.
    
    Paper: "Deep Networks with Stochastic Depth" (Huang et al., ECCV 2016)
    
    Examples of drop_prob: typically 0.1-0.3 for ViT/Swin models
    """
    def __init__(self, drop_prob=None, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=False):
        """
        Args:
            x: Input tensor of any shape
            training: Boolean indicating training mode
        
        Returns:
            Output with dropped residual paths during training
        """
        if not training or self.drop_prob == 0.0:
            return x
        
        # Generate binary mask with shape (batch_size, 1, 1, ..., 1)
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)  # Broadcast mask to all spatial dims
        
        # Create random mask: values > drop_prob survive
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        
        # Scale surviving paths to maintain expected value
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output


class WindowAttention(layers.Layer):
    """
    Window-based Multi-Head Self Attention (W-MSA) with Relative Position Bias.
    
    This is the core attention mechanism of Swin Transformer. Unlike standard
    multi-head attention that attends to all tokens (O(n²) complexity), window
    attention only attends within local windows, reducing complexity to O(n).
    
    Key Features:
    1. Local attention within 7x7 windows (reduces from O(HW) to O(7²))
    2. Relative position bias: learnable bias based on relative positions
       (not absolute), which improves translation equivariance
    3. Query, Key, Value projections with optional bias terms
    
    Mathematical Details:
    - Attention = Softmax(Q @ K^T / sqrt(d) + RelPosBias) @ V
    - RelPosBias is query-independent (computed once per head)
    
    For XAI Research:
    The attention weights (attn) can be extracted and visualized to understand
    which image regions the model focuses on during classification.
    """
    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs
    ):
        """
        Args:
            dim: Total embedding dimension (must be divisible by num_heads)
            window_size: Tuple (window_h, window_w) for the attention window
            num_heads: Number of parallel attention heads
            qkv_bias: Whether to use bias in Query, Key, Value projections
            dropout_rate: Dropout rate for attention weights
        """
        super(WindowAttention, self).__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5  # Scaling factor: 1/sqrt(head_dim)
        
        # Project input tokens to Q, K, V (all in one for efficiency)
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        
        # Project concatenated attention outputs back to original dimension
        self.proj = layers.Dense(dim)
        
        # Pre-compute relative position indices (eager evaluation in __init__)
        # This maps each (query_token, key_token) pair to an index in the bias table
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.reshape(2, -1)
        
        # Calculate relative coordinates: (query_position - key_position)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        
        # Shift indices to positive range: add (window_size - 1) so indices are 0 to (2*window_size - 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        
        # Flatten 2D indices to 1D using row-major order
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        
        # Store as numpy array for efficient lookup (not a trainable weight)
        self.relative_position_index_np = relative_position_index.astype('int32')

    def build(self, input_shape):
        """
        Build relative position bias table.
        
        Relative position bias is a key innovation in Swin Transformer:
        - Instead of absolute positional encoding, we use relative position biases
        - This makes the model translation-equivariant (shifts don't change relative positions)
        - The bias is query-independent (computed once per head per input)
        
        For a 7x7 window, relative offsets range from -6 to +6 in both dimensions,
        giving (2*7-1) × (2*7-1) = 13 × 13 = 169 unique relative position pairs.
        """
        # Calculate number of unique relative position pairs
        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )
        
        # Learnable relative position bias table: (num_relative_positions, num_heads)
        # Each head has its own bias preferences for different relative positions
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),  # He init for stability
            trainable=True,
            name="relative_position_bias_table",
        )

    def call(self, x, mask=None):
        """
        Compute window-based multi-head self attention.
        
        Args:
            x: Input tokens of shape (num_windows, window_size², channels)
               Example: (10, 49, 768) for 10 windows, each 7×7=49 tokens
            mask: Attention mask for shifted windows (used for cross-window prevention)
        
        Returns:
            Attended output of shape (num_windows, window_size², channels)
        """
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        
        # Linear projection to Q, K, V (all at once: dim*3 → split into 3 parts)
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))  # (3, batch_windows, heads, size, head_dim)
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        
        # Compute attention scores with scaling
        q = q * self.scale  # Scale by 1/sqrt(head_dim) for numerical stability
        k = tf.transpose(k, perm=(0, 1, 3, 2))  # (batch_windows, heads, head_dim, size)
        attn = q @ k  # (batch_windows, heads, size, size) - attention logits

        # Add learnable relative position bias
        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(
            tf.constant(self.relative_position_index_np), shape=(-1,)
        )
        # Lookup relative position bias from the table
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, relative_position_index_flat
        )
        # Reshape to (window_size², window_size², num_heads)
        relative_position_bias = tf.reshape(
            relative_position_bias, shape=(num_window_elements, num_window_elements, -1)
        )
        # Transpose to (num_heads, window_size², window_size²)
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        # Add bias to attention scores
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        # Apply mask (prevents attention across different regions in shifted windows)
        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = (
                tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size))
                + mask_float  # Add strong negative values (-100) to masked positions
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            # No mask: standard softmax normalization
            attn = keras.activations.softmax(attn, axis=-1)
        
        attn = self.dropout(attn)  # Dropout for regularization

        # Apply attention to values
        x_qkv = attn @ v  # (batch_windows, heads, size, head_dim)
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))  # (batch_windows, size, heads, head_dim)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))  # Concat heads back to full dimension
        
        # Final output projection
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv


class SwinTransformerBlock(layers.Layer):
    """
    Swin Transformer Block - The fundamental building block of the architecture.
    
    This block combines:
    1. Window-based Multi-Head Self Attention (W-MSA)
    2. Shifted Window-based Multi-Head Self Attention (SW-MSA)
    3. Feed-Forward Network (FFN / MLP)
    
    The Shifted Windows are crucial for enabling cross-window communication:
    - In even-indexed blocks, we use standard window attention (no shift)
    - In odd-indexed blocks, we shift the window by (window_size // 2) pixels
    
    This alternating pattern allows different regions to interact while maintaining
    computational efficiency. An attention mask prevents attending to shifted-out regions.
    
    Architecture Flow:
    1. Input → LayerNorm → W-MSA/SW-MSA → DropPath + Residual
    2. → LayerNorm → FFN → DropPath + Residual → Output
    
    For XAI Research:
    Store attention weights from self.attn to visualize which image regions
    the model attends to at each layer and stage.
    """
    def __init__(
        self,
        dim,
        num_patch,
        num_heads,
        window_size=7,
        shift_size=0,
        num_mlp=1024,
        qkv_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        """
        Args:
            dim: Input/output embedding dimension
            num_patch: Tuple (height_patches, width_patches) - spatial resolution in patches
            num_heads: Number of attention heads
            window_size: Size of local attention windows
            shift_size: Amount to shift windows (0 for regular, >0 for shifted attention)
            num_mlp: Hidden dimension of the feedforward network
            qkv_bias: Whether to use bias in Q, K, V projections
            dropout_rate: Dropout rate for regularization
        """
        super(SwinTransformerBlock, self).__init__(**kwargs)
        self.dim = dim
        self.num_patch = num_patch
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_mlp = num_mlp

        # First sub-layer: Layernorm + Window Attention
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = DropPath(dropout_rate)
        
        # Second sub-layer: Layernorm + MLP (Feed-Forward Network)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = keras.Sequential(
            [
                layers.Dense(num_mlp),
                layers.Activation(keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),  # Project back to original dimension
                layers.Dropout(dropout_rate),
            ]
        )

        # Safety check: if patches are smaller than window, adjust window size
        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)
        
        # Pre-compute attention mask for shifted windows (eager evaluation in __init__)
        # This prevents attention between regions that should be disconnected after the shift
        if self.shift_size == 0:
            self.attn_mask_np = None
        else:
            height, width = self.num_patch
            
            # Define 3x3 regions after shifting (using numpy)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            
            # Create mask array with region IDs (pure numpy)
            mask_array = np.zeros((height, width))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[h, w] = count
                    count += 1
            
            # Partition mask into windows manually (numpy version)
            num_h = height // self.window_size
            num_w = width // self.window_size
            mask_windows = []
            for i in range(num_h):
                for j in range(num_w):
                    h_start = i * self.window_size
                    w_start = j * self.window_size
                    window = mask_array[h_start:h_start+self.window_size, 
                                       w_start:w_start+self.window_size]
                    mask_windows.append(window.flatten())
            mask_windows = np.array(mask_windows)  # (num_windows, window_size²)
            
            # Create attention mask: where region IDs differ, set attention to -100
            attn_mask = mask_windows[:, :, None] - mask_windows[:, None, :]  # broadcast difference
            attn_mask = np.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = np.where(attn_mask == 0, 0.0, attn_mask)
            
            # Store as numpy array for efficient lookup
            self.attn_mask_np = attn_mask.astype('float32')

    def call(self, x):
        """
        Forward pass through Swin Transformer Block.
        
        Steps:
        1. First sub-layer: Residual + Dropout
           - Apply LayerNorm
           - Apply [Shifted] Window-based Multi-Head Attention
           - Apply DropPath (stochastic depth)
           - Add residual connection
        
        2. Second sub-layer: Residual + Dropout
           - Apply LayerNorm
           - Apply MLP (feed-forward network with 4x expansion)
           - Apply DropPath (stochastic depth)
           - Add residual connection
        
        Args:
            x: Input patch sequence of shape (batch, num_patches, dim)
        
        Returns:
            Output patch sequence of shape (batch, num_patches, dim)
        """
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        
        # ===== First Sub-layer: Attention =====
        x_skip = x  # Save for residual connection
        x = self.norm1(x)  # Pre-norm (LayerNorm before attention)
        x = tf.reshape(x, shape=(-1, height, width, channels))  # Reshape to spatial
        
        # Shifted window attention (SW-MSA) or Normal window attention (W-MSA)
        if self.shift_size > 0:
            # Roll/shift the window (cyclically pad to maintain size)
            # Shift of 3 means positions wrap around: [0,1,2,3,4,5,6] → [3,4,5,6,0,1,2]
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            # No shift (regular window attention)
            shifted_x = x

        # Partition into windows for efficient attendance
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=(-1, self.window_size * self.window_size, channels)
        )
        
        # Apply window attention (with mask for SW-MSA)
        mask_tensor = tf.constant(self.attn_mask_np) if self.attn_mask_np is not None else None
        attn_windows = self.attn(x_windows, mask=mask_tensor)

        # Reconstruct from windows
        attn_windows = tf.reshape(
            attn_windows, shape=(-1, self.window_size, self.window_size, channels)
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        
        # Roll back if we shifted
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        # Reshape back to patch sequence and apply residual connection
        x = tf.reshape(x, shape=(-1, height * width, channels))
        x = self.drop_path(x)  # Stochastic depth
        x = x_skip + x  # Residual connection
        
        # ===== Second Sub-layer: MLP (Feed-Forward Network) =====
        x_skip = x  # Save for residual connection
        x = self.norm2(x)  # Pre-norm
        x = self.mlp(x)  # Feed-forward (dense → GELU → dense)
        x = self.drop_path(x)  # Stochastic depth
        x = x_skip + x  # Residual connection
        
        return x


class PatchExtract(layers.Layer):
    """
    Extract non-overlapping patches from input images.
    
    This is the first step of the Swin architecture. Converts a continuous 2D image
    into a sequence of fixed-size patches, which are then treated as tokens.
    
    For example, a 128×128 image with 4×4 patches becomes 32×32=1024 patches.
    Each patch has 4×4×3 = 48 values (for RGB images).
    
    This dramatically reduces computational complexity compared to ViT or CNN:
    - CNN uses sliding windows (overlapping, expensive)
    - ViT uses all-to-all attention (O(n²) where n = num_patches)
    - Swin uses window attention (O(n) with local windows)
    """
    def __init__(self, patch_size, **kwargs):
        """
        Args:
            patch_size: Tuple (patch_h, patch_w) - size of square patches
        """
        super(PatchExtract, self).__init__(**kwargs)
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[1]

    def call(self, images):
        """
        Extract patches from images.
        
        Args:
            images: Input image tensor of shape (batch, height, width, channels)
        
        Returns:
            Flattened patches of shape (batch, num_patches, patch_height*patch_width*channels)
            where num_patches = (height // patch_size) * (width // patch_size)
        """
        batch_size = tf.shape(images)[0]
        
        # Extract patches using sliding window (with stride = patch_size for non-overlapping)
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size_x, self.patch_size_y, 1),
            strides=(1, self.patch_size_x, self.patch_size_y, 1),  # Non-overlapping
            rates=(1, 1, 1, 1),
            padding="VALID",  # No padding
        )
        
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))


class PatchEmbedding(layers.Layer):
    """
    Embed extracted patches and add positional information.
    
    After patch extraction, each patch is a flat vector (e.g., 4×4×3=48 values).
    This layer projects each patch to the embedding dimension and adds positional
    encodings to preserve spatial information.
    
    The positional embeddings are learned (not sinusoidal like in ViT), which works
    well for Swin's hierarchical architecture with local windows.
    """
    def __init__(self, num_patch, embed_dim, **kwargs):
        """
        Args:
            num_patch: Total number of patches (height_patches * width_patches)
            embed_dim: Target embedding dimension
        """
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)  # Linear projection to embedding dimension
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)  # Learned pos embeddings

    def call(self, patch):
        """
        Args:
            patch: Flattened patches of shape (batch, num_patches, patch_dim)
        
        Returns:
            Embedded patches with positional information of shape (batch, num_patches, embed_dim)
        """
        # Create positional indices [0, 1, 2, ..., num_patches-1]
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        # Project patches to embedding dimension and add position embeddings
        return self.proj(patch) + self.pos_embed(pos)


class PatchMerging(layers.Layer):
    """
    Hierarchical Patch Merging and Dimension Expansion.
    
    Used at stage transitions to achieve hierarchical representation:
    - Merges 2×2 neighboring patches into one (2x spatial downsampling)
    - Expands embedding dimension by 4x (concatenate 4 patches → 4x channels)
    - Applies linear transformation to reach target dimension
    
    Example:
    - Stage 1 end: 56×56 patches, 192 channels
    - After merging: 28×28 patches, 384 channels (192*2=384)
    - Stage 2 start: 28×28 patches, 384 channels
    
    This hierarchical downsampling creates a pyramid structure like CNNs,
    which improves performance on dense prediction tasks.
    """
    def __init__(self, num_patch, embed_dim, **kwargs):
        """
        Args:
            num_patch: Tuple (height_patches, width_patches) of current resolution
            embed_dim: Current embedding dimension
        """
        super(PatchMerging, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        # Linear projection: (4 * embed_dim) → (2 * embed_dim)
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)

    def call(self, x):
        """
        Merge patches and expand representation.
        
        Steps:
        1. Reshape from (batch, num_patches, channels) to (batch, height, width, channels)
        2. Extract 2×2 non-overlapping groups: top-left, top-right, bottom-left, bottom-right
        3. Concatenate all 4 groups along channel dimension (4x expansion)
        4. Apply linear transformation to compress to target dimension
        
        Example with 2×2 merging:
        Original: 4×4 spatial grid (16 patches)
        After merge: 2×2 spatial grid (4 patches)
        Each new patch = concat of 4 old patches: 4*C channels
        Linear transform reduces: 4*C → 2*C channels
        
        Args:
            x: Patch sequence of shape (batch, num_patches, channels)
               where num_patches = height * width
        
        Returns:
            Merged patches of shape (batch, num_patches//4, 2*channels)
        """
        height, width = self.num_patch
        _, _, C = x.get_shape().as_list()
        
        # Reshape to spatial dimensions
        x = tf.reshape(x, shape=(-1, height, width, C))
        
        # Extract alternating pixels (every 2nd element in both dimensions)
        x0 = x[:, 0::2, 0::2, :]   # Top-left quadrant
        x1 = x[:, 1::2, 0::2, :]   # Bottom-left quadrant
        x2 = x[:, 0::2, 1::2, :]   # Top-right quadrant
        x3 = x[:, 1::2, 1::2, :]   # Bottom-right quadrant
        
        # Concatenate all 4 groups: (batch, H/2, W/2, 4*C)
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        
        # Reshape to patch sequence and apply linear projection
        x = tf.reshape(x, shape=(-1, (height // 2) * (width // 2), 4 * C))
        return self.linear_trans(x)


# ========== MAIN ARCHITECTURE ==========

def build_swin_tiny(input_shape, num_classes=2):
    """
    Build Swin-Tiny model - lightweight hierarchical vision transformer for deepfake detection.
    
    Architecture Hierarchy:
    - Stage 1 (2 blocks):   embedding_dim=96,  heads=3,  spatial_size = input_size / 4
    - Stage 2 (2 blocks):   embedding_dim=192, heads=6,  spatial_size = input_size / 8
    - Stage 3 (6 blocks):   embedding_dim=384, heads=12, spatial_size = input_size / 16 [DEEPEST]
    - Stage 4 (2 blocks):   embedding_dim=768, heads=24, spatial_size = input_size / 32
    
    Total Parameters: ~28M (8x smaller than Swin-Large, ideal for resource-constrained scenarios)
    
    Key Design Choices:
    1. 4x4 patch embedding (matches original Swin-Tiny)
    2. Hierarchical downsampling via patch merging (like ResNet pyramid)
    3. Shifted windows for efficient cross-region communication
    4. Relative position bias for translation equivariance
    5. Pre-norm architecture for training stability
    
    Advantages of Swin-Tiny:
    - Fast training on single GPU
    - Suitable for edge deployment
    - Quick experimentation and prototyping
    - Still maintains strong performance on deepfake detection tasks
    
    For XAI Research:
    - Attention weights can be extracted at each stage for visualization
    - Features become progressively deeper and more semantic
    - Stage 3 (deepest with 6 blocks) usually contains discriminative deepfake features
    
    Args:
        input_shape: Tuple (height, width, channels) - e.g., (128, 128, 3) or (175, 175, 3)
        num_classes: Number of output classes (default: 2 for binary deepfake detection)
    
    Returns:
        Keras Model ready for training/inference
    """
    inputs = layers.Input(shape=input_shape)
    
    # ===== STAGE 0: Patch Extraction and Initial Embedding =====
    # Convert image to patch sequence and add positional information
    num_patch_x = input_shape[0] // patch_size[0]
    num_patch_y = input_shape[1] // patch_size[1]
    num_patch = num_patch_x * num_patch_y
    
    # Extract non-overlapping patches: (B, H, W, C) → (B, num_patches, patch_dim)
    x = PatchExtract(patch_size)(inputs)
    # Project patches to embedding dimension: (B, num_patches, D1)
    x = PatchEmbedding(num_patch, embed_dim_stages[0])(x)
    
    # ===== STAGE 1: Shallow Feature Extraction =====
    # 2 Swin Blocks with 3 attention heads, embed_dim=96
    # Processes: spatial_size = input_size / 4
    # Purpose: Extract low-level features (edges, textures, colors)
    for i in range(depths[0]):
        x = SwinTransformerBlock(
            dim=embed_dim_stages[0],
            num_patch=(num_patch_x, num_patch_y),
            num_heads=num_heads_stages[0],
            window_size=window_size,
            shift_size=shift_size if i % 2 == 1 else 0,  # Alternate: no shift, then shift
            num_mlp=num_mlp_stages[0],
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
            name=f"swin_block_s1_{i}",
        )(x)
    
    # Hierarchical downsampling: merge 2×2 patches into 1
    # Increases dimension: 192 → 384 (doubled)
    # Reduces spatial size: 2x downsampling
    x = PatchMerging((num_patch_x, num_patch_y), embed_dim_stages[0])(x)
    num_patch_x //= 2
    num_patch_y //= 2
    
    # ===== STAGE 2: Intermediate Feature Extraction =====
    # 2 Swin Blocks with 6 attention heads, embed_dim=192
    # Processes: spatial_size = input_size / 8
    # Purpose: Extract mid-level features (object parts, shapes)
    for i in range(depths[1]):
        x = SwinTransformerBlock(
            dim=embed_dim_stages[1],
            num_patch=(num_patch_x, num_patch_y),
            num_heads=num_heads_stages[1],
            window_size=window_size,
            shift_size=shift_size if i % 2 == 1 else 0,
            num_mlp=num_mlp_stages[1],
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
            name=f"swin_block_s2_{i}",
        )(x)
    
    # Hierarchical downsampling
    x = PatchMerging((num_patch_x, num_patch_y), embed_dim_stages[1])(x)
    num_patch_x //= 2
    num_patch_y //= 2
    
    # ===== STAGE 3: Deep Feature Extraction (MAIN ENCODER) =====
    # 6 Swin Blocks (DEEPEST STAGE) with 12 attention heads, embed_dim=384
    # Processes: spatial_size = input_size / 16
    # Purpose: Extract high-level semantic features
    # NOTE: This stage is where discriminative deepfake features form
    #       Great candidate for XAI visualization and attention analysis
    for i in range(depths[2]):
        x = SwinTransformerBlock(
            dim=embed_dim_stages[2],
            num_patch=(num_patch_x, num_patch_y),
            num_heads=num_heads_stages[2],
            window_size=window_size,
            shift_size=shift_size if i % 2 == 1 else 0,
            num_mlp=num_mlp_stages[2],
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
            name=f"swin_block_s3_{i}",  # For XAI: extract attention from these layers
        )(x)
    
    # Hierarchical downsampling
    x = PatchMerging((num_patch_x, num_patch_y), embed_dim_stages[2])(x)
    num_patch_x //= 2
    num_patch_y //= 2
    
    # ===== STAGE 4: Final Feature Refinement =====
    # 2 Swin Blocks with 24 attention heads, embed_dim=768 (HIGHEST DIMENSION)
    # Processes: spatial_size = input_size / 32
    # Purpose: Refine features for final classification decision
    for i in range(depths[3]):
        x = SwinTransformerBlock(
            dim=embed_dim_stages[3],
            num_patch=(num_patch_x, num_patch_y),
            num_heads=num_heads_stages[3],
            window_size=window_size,
            shift_size=shift_size if i % 2 == 1 else 0,
            num_mlp=num_mlp_stages[3],
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
            name=f"swin_block_s4_{i}",
        )(x)
    
    # ===== CLASSIFICATION HEAD =====
    # Apply final layer normalization
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    
    # Global average pooling: reduce spatial dimensions to 1
    # (B, 8, 8, 768) → (B, 768) for 128×128 input
    x = layers.GlobalAveragePooling1D()(x)
    
    # Final classification layer
    # Maps high-dimensional features to class probabilities
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create and return the model
    model = keras.Model(inputs=inputs, outputs=outputs, name="SwinTiny_XAI_Research")
    return model