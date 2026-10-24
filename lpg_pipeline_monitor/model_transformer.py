"""
model_transformer.py — Temporal Transformer dual-head for LPG pipeline fault detection.

Architecture
    Input (batch, T, F)
      → Linear projection to d_model
      → Sinusoidal Positional Encoding
      → N × Transformer Encoder Block:
            MultiHeadSelfAttention (GELU FFN) + LayerNorm + residual
      → GlobalAveragePooling1D
      → Shared Dense (128 → 64)
      → fault_type head   (3-class softmax)
      → severity_label head (3-class softmax)

Why Transformer for fault detection?
  • Self-attention captures arbitrary-range pairwise relationships between
    timesteps (e.g., pressure-drop at t=2 correlated with flow-rise at t=8)
    without inductive bias toward adjacent steps, which LSTM can miss.
  • GELU activation in FFN provides smoother gradients than ReLU.
  • Positional encoding preserves temporal order without hard-coding
    recurrence; the model can attend globally from the first layer.

Usage:
    from model_transformer import build_transformer_model
    model = build_transformer_model()            # 6 raw features
    model = build_transformer_model(n_features=12)  # raw + delta
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers

from config import (
    SEQ_LEN, N_FEATURES,
    N_FAULT_CLASSES, N_LABEL_CLASSES,
    LEARNING_RATE,
)


# ── Custom layer: sinusoidal positional encoding ──────────────────────────────

@tf.keras.utils.register_keras_serializable(package='lpg_monitor')
class PositionalEncoding(tf.keras.layers.Layer):
    """Fixed sinusoidal positional encoding added to input embeddings.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, max_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

        pos    = np.arange(max_len)[:, np.newaxis]          # (max_len, 1)
        i      = np.arange(d_model)[np.newaxis, :]          # (1, d_model)
        angles = pos / np.power(10000.0, (2 * (i // 2)) / float(d_model))
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        # Shape (1, max_len, d_model) — broadcast over batch
        self._pe = tf.constant(angles[np.newaxis], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self._pe[:, :seq_len, :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'max_len': self.max_len, 'd_model': self.d_model})
        return cfg


# ── Transformer encoder block ─────────────────────────────────────────────────

def _encoder_block(x, d_model: int, num_heads: int, dff: int,
                   dropout_rate: float, l2, prefix: str):
    """One Transformer encoder block with pre-norm residual layout."""
    # ── Multi-head self-attention ──────────────────────────────────────────
    attn_in = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name=f'{prefix}_ln1')(x)
    attn_out = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout_rate,
        name=f'{prefix}_mha',
    )(attn_in, attn_in)
    attn_out = tf.keras.layers.Dropout(
        dropout_rate, name=f'{prefix}_attn_drop')(attn_out)
    x = tf.keras.layers.Add(name=f'{prefix}_res1')([x, attn_out])

    # ── Feed-forward network ──────────────────────────────────────────────
    ffn_in = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name=f'{prefix}_ln2')(x)
    ffn = tf.keras.layers.Dense(
        dff, activation='gelu', kernel_regularizer=l2,
        name=f'{prefix}_ffn1')(ffn_in)
    ffn = tf.keras.layers.Dense(
        d_model, kernel_regularizer=l2, name=f'{prefix}_ffn2')(ffn)
    ffn = tf.keras.layers.Dropout(
        dropout_rate, name=f'{prefix}_ffn_drop')(ffn)
    x = tf.keras.layers.Add(name=f'{prefix}_res2')([x, ffn])

    return x


# ── Model factory ─────────────────────────────────────────────────────────────

def build_transformer_model(
    seq_len:         int   = SEQ_LEN,
    n_features:      int   = N_FEATURES,
    n_fault_classes: int   = N_FAULT_CLASSES,
    n_label_classes: int   = N_LABEL_CLASSES,
    lr:              float = LEARNING_RATE,
    d_model:         int   = 64,
    num_heads:       int   = 4,
    dff:             int   = 256,
    num_layers:      int   = 4,
    dropout_rate:    float = 0.20,
    l2_reg:          float = 0.0005,
) -> tf.keras.Model:
    """
    Build and compile the Temporal Transformer dual-output model.

    Parameters
    ----------
    d_model      : embedding dimension (must be divisible by num_heads)
    num_heads    : number of attention heads
    dff          : feed-forward network hidden dimension
    num_layers   : number of stacked encoder blocks
    dropout_rate : dropout applied to attention weights and FFN outputs
    l2_reg       : L2 regularization coefficient

    Returns
    -------
    model : tf.keras.Model
        Compiled model with two outputs:
          'fault_type'     → sparse categorical crossentropy  (weight 1.0)
          'severity_label' → sparse categorical crossentropy  (weight 0.4)
    """
    l2  = regularizers.l2(l2_reg)
    inp = tf.keras.Input(shape=(seq_len, n_features), name='sensor_input')

    # ── Linear input projection to d_model ───────────────────────────────────
    x = tf.keras.layers.Dense(d_model, name='input_proj')(inp)

    # ── Add positional encoding ───────────────────────────────────────────────
    x = PositionalEncoding(max_len=seq_len, d_model=d_model,
                            name='pos_enc')(x)
    x = tf.keras.layers.Dropout(dropout_rate, name='drop_pos')(x)

    # ── Stack Transformer encoder blocks ─────────────────────────────────────
    for i in range(num_layers):
        x = _encoder_block(x, d_model, num_heads, dff, dropout_rate, l2,
                            prefix=f'enc_{i}')

    # ── Final layer norm (post-stack) ─────────────────────────────────────────
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='final_ln')(x)

    # ── Pooling: aggregate T timesteps → single vector ────────────────────────
    x = tf.keras.layers.GlobalAveragePooling1D(name='gap')(x)

    # ── Shared dense representation ───────────────────────────────────────────
    shared = tf.keras.layers.Dense(
        128, activation='relu', kernel_regularizer=l2, name='dense_shared')(x)
    shared = tf.keras.layers.Dropout(0.25, name='drop_shared')(shared)
    shared = tf.keras.layers.Dense(
        64, activation='relu', kernel_regularizer=l2, name='dense_shared2')(shared)

    # ── Head 1: Fault Type (none / leak / blockage) ────────────────────────────
    fault_branch = tf.keras.layers.Dense(
        32, activation='relu', kernel_regularizer=l2, name='fault_branch')(shared)
    fault_out = tf.keras.layers.Dense(
        n_fault_classes, activation='softmax', name='fault_type')(fault_branch)

    # ── Head 2: Severity Label (Normal / Warning / Danger) ────────────────────
    label_branch = tf.keras.layers.Dense(
        32, activation='relu', kernel_regularizer=l2, name='label_branch')(shared)
    label_out = tf.keras.layers.Dense(
        n_label_classes, activation='softmax', name='severity_label')(label_branch)

    # ── Compile ───────────────────────────────────────────────────────────────
    model = tf.keras.Model(
        inputs=inp,
        outputs=[fault_out, label_out],
        name='Temporal_Transformer_LPG',
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss={
            'fault_type':     'sparse_categorical_crossentropy',
            'severity_label': 'sparse_categorical_crossentropy',
        },
        loss_weights={
            'fault_type':     1.0,
            'severity_label': 0.4,
        },
        metrics={
            'fault_type':     ['accuracy'],
            'severity_label': ['accuracy'],
        },
    )
    return model


if __name__ == '__main__':
    m = build_transformer_model()
    m.summary()
    print(f"\nTotal trainable parameters: {m.count_params():,}")
