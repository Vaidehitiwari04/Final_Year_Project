"""
model.py — CNN-LSTM v2: improved dual-head architecture with residual connections,
           LayerNormalization, and L2 regularization to reduce overfitting.

Changes vs v1
  ● Causal Conv blocks now have residual (skip) connections
  ● BatchNorm replaced by LayerNorm (more stable for time-series, works with
    variable batch sizes and tf.data)
  ● L2 weight-decay (0.0005) on all learnable layers
  ● Increased dropout to 0.40 in conv & LSTM towers
  ● recurrent_dropout=0.10 on LSTM gates (Variational Dropout)
  ● Adam with clipnorm=1.0 to prevent exploding gradients

Usage:
    from model import build_model
    model = build_model()          # 6 raw features (default)
    model = build_model(n_features=12)   # raw + delta feature set
"""

import tensorflow as tf
from tensorflow.keras import regularizers

from config import (
    SEQ_LEN, N_FEATURES,
    N_FAULT_CLASSES, N_LABEL_CLASSES,
    LEARNING_RATE,
)


def build_model(
    seq_len:         int   = SEQ_LEN,
    n_features:      int   = N_FEATURES,
    n_fault_classes: int   = N_FAULT_CLASSES,
    n_label_classes: int   = N_LABEL_CLASSES,
    lr:              float = LEARNING_RATE,
    dropout_rate:    float = 0.40,
    l2_reg:          float = 0.0005,
) -> tf.keras.Model:
    """
    Build and compile the improved CNN-LSTM dual-output model.

    Returns
    -------
    model : tf.keras.Model
        Compiled with two outputs:
          'fault_type'     → sparse categorical crossentropy  (weight 1.0)
          'severity_label' → sparse categorical crossentropy  (weight 0.4)
    """
    l2 = regularizers.l2(l2_reg)
    inp = tf.keras.Input(shape=(seq_len, n_features), name='sensor_input')

    # ── Block 1: Causal Conv residual block ───────────────────────────────────
    # First layer: project input channels → 64
    x = tf.keras.layers.Conv1D(
        64, kernel_size=3, padding='causal', activation='relu',
        kernel_regularizer=l2, name='conv1a')(inp)
    x = tf.keras.layers.LayerNormalization(name='ln1a')(x)

    # Second layer within residual block
    x1 = tf.keras.layers.Conv1D(
        64, kernel_size=3, padding='causal', activation='relu',
        kernel_regularizer=l2, name='conv1b')(x)
    x1 = tf.keras.layers.LayerNormalization(name='ln1b')(x1)
    x = tf.keras.layers.Add(name='res_add1')([x, x1])      # skip connection

    # ── Block 2: Deeper causal conv ───────────────────────────────────────────
    x2 = tf.keras.layers.Conv1D(
        128, kernel_size=3, padding='causal', activation='relu',
        kernel_regularizer=l2, name='conv2a')(x)
    x2 = tf.keras.layers.LayerNormalization(name='ln2a')(x2)

    # 1×1 conv to project back to 64 channels for the residual add
    x2 = tf.keras.layers.Conv1D(
        64, kernel_size=1, padding='causal', activation='relu',
        kernel_regularizer=l2, name='conv2b_proj')(x2)
    x2 = tf.keras.layers.LayerNormalization(name='ln2b')(x2)
    x = tf.keras.layers.Add(name='res_add2')([x, x2])

    x = tf.keras.layers.Dropout(dropout_rate, name='drop_cnn')(x)

    # ── Block 3: Bidirectional LSTM ───────────────────────────────────────────
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            64, return_sequences=True,
            kernel_regularizer=l2, recurrent_dropout=0.10),
        name='bilstm_1')(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            32, return_sequences=False,
            kernel_regularizer=l2, recurrent_dropout=0.10),
        name='bilstm_2')(x)
    x = tf.keras.layers.Dropout(dropout_rate, name='drop_lstm')(x)

    # ── Shared dense representation ───────────────────────────────────────────
    shared = tf.keras.layers.Dense(
        128, activation='relu', kernel_regularizer=l2, name='dense_shared')(x)
    shared = tf.keras.layers.Dropout(0.30, name='drop_shared')(shared)
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
        name='CNN_LSTM_v2_LPG',
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
    m = build_model()
    m.summary()
    print(f"\nTotal trainable parameters: {m.count_params():,}")
