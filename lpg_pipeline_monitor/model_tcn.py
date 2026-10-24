"""
model_tcn.py — Temporal Convolutional Network (TCN) dual-head for LPG fault detection.

Architecture
    Input (batch, T, F)
      → Stack of N dilated residual causal-conv blocks
          (dilation rates: 1, 2, 4, 8 — exponentially growing receptive field)
      → GlobalAveragePooling1D
      → Shared Dense (128 → 64)
      → fault_type head   (3-class softmax)
      → severity_label head (3-class softmax)

TCN Receptive Field (with kernel_size=3, dilations=[1,2,4,8]):
    RF = 1 + 2 × (kernel_size − 1) × Σ(dilation)
       = 1 + 2 × 2 × (1+2+4+8) = 61 timesteps
    → Covers the full 50-second window (T=10 steps) with room to spare.

Key properties:
  • Parallelisable (no sequential RNN dependency)
  • Stable gradients via residual connections
  • SpatialDropout1D regularises entire feature maps (better than scalar dropout
    for convolutional sequences)

Usage:
    from model_tcn import build_tcn_model
    model = build_tcn_model()           # 6 raw features
    model = build_tcn_model(n_features=12)  # raw + delta
"""

import tensorflow as tf
from tensorflow.keras import regularizers

from config import (
    SEQ_LEN, N_FEATURES,
    N_FAULT_CLASSES, N_LABEL_CLASSES,
    LEARNING_RATE,
)


# ── Dilated residual block ─────────────────────────────────────────────────────

def _tcn_residual_block(
    x,
    filters:       int,
    kernel_size:   int,
    dilation_rate: int,
    dropout_rate:  float,
    l2,
    prefix:        str,
):
    """Two dilated causal convolutions with a skip connection.

    Each conv is followed by LayerNorm + ReLU + SpatialDropout1D.
    If input channel count ≠ filters, a 1×1 conv projects the residual.
    """
    residual = x

    # ── First dilated conv ────────────────────────────────────────────────
    x = tf.keras.layers.Conv1D(
        filters, kernel_size,
        padding='causal',
        dilation_rate=dilation_rate,
        kernel_regularizer=l2,
        name=f'{prefix}_c1',
    )(x)
    x = tf.keras.layers.LayerNormalization(name=f'{prefix}_ln1')(x)
    x = tf.keras.layers.Activation('relu', name=f'{prefix}_act1')(x)
    x = tf.keras.layers.SpatialDropout1D(dropout_rate, name=f'{prefix}_drop1')(x)

    # ── Second dilated conv ───────────────────────────────────────────────
    x = tf.keras.layers.Conv1D(
        filters, kernel_size,
        padding='causal',
        dilation_rate=dilation_rate,
        kernel_regularizer=l2,
        name=f'{prefix}_c2',
    )(x)
    x = tf.keras.layers.LayerNormalization(name=f'{prefix}_ln2')(x)
    x = tf.keras.layers.Activation('relu', name=f'{prefix}_act2')(x)
    x = tf.keras.layers.SpatialDropout1D(dropout_rate, name=f'{prefix}_drop2')(x)

    # ── Residual skip (with optional channel projection) ──────────────────
    if residual.shape[-1] != filters:
        residual = tf.keras.layers.Conv1D(
            filters, 1, padding='same',
            name=f'{prefix}_proj',
        )(residual)

    return tf.keras.layers.Add(name=f'{prefix}_add')([x, residual])


# ── Model factory ─────────────────────────────────────────────────────────────

def build_tcn_model(
    seq_len:         int   = SEQ_LEN,
    n_features:      int   = N_FEATURES,
    n_fault_classes: int   = N_FAULT_CLASSES,
    n_label_classes: int   = N_LABEL_CLASSES,
    lr:              float = LEARNING_RATE,
    filters:         int   = 64,
    kernel_size:     int   = 3,
    dilations:       tuple = (1, 2, 4, 8),
    dropout_rate:    float = 0.20,
    l2_reg:          float = 0.0005,
) -> tf.keras.Model:
    """
    Build and compile the TCN dual-output model.

    Parameters
    ----------
    filters      : number of convolutional filters in every TCN block
    kernel_size  : causal conv kernel width
    dilations    : sequence of dilation rates; one block per entry
    dropout_rate : SpatialDropout1D rate applied after each conv
    l2_reg       : L2 regularization coefficient

    Returns
    -------
    model : tf.keras.Model
        Compiled with two outputs:
          'fault_type'     → sparse categorical crossentropy  (weight 1.0)
          'severity_label' → sparse categorical crossentropy  (weight 0.4)
    """
    l2  = regularizers.l2(l2_reg)
    inp = tf.keras.Input(shape=(seq_len, n_features), name='sensor_input')
    x   = inp

    # ── Stack dilated residual blocks ─────────────────────────────────────────
    for i, dilation in enumerate(dilations):
        x = _tcn_residual_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            dropout_rate=dropout_rate,
            l2=l2,
            prefix=f'tcn_{i}_d{dilation}',
        )

    # ── Global average pooling ────────────────────────────────────────────────
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
        name='TCN_LPG_Monitor',
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
    m = build_tcn_model()
    m.summary()
    print(f"\nTotal trainable parameters: {m.count_params():,}")
    rf = 1 + 2 * (3 - 1) * sum([1, 2, 4, 8])
    print(f"Receptive field: {rf} timesteps")
