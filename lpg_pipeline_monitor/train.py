"""
train.py — Training pipeline for the CNN-LSTM LPG fault detection model.

Run:
    cd lpg_pipeline_monitor
    python train.py

Saves:
    models/best_model.keras   — best checkpoint (monitored: val_fault_type_accuracy)
    results/training_history.json  — full epoch-by-epoch metrics
"""

import os
import json
import time
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
import tensorflow as tf

from config import (
    SPLITS_DIR, MODELS_DIR, RESULTS_DIR,
    MODEL_PATH, HISTORY_PATH,
    BATCH_SIZE, EPOCHS, PATIENCE, LR_PATIENCE, LEARNING_RATE,
    FAULT_TYPE_LABELS, SEVERITY_MAP,
)
from model import build_model


# ── Reproducibility ───────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)


def load_splits():
    """Load all pre-processed numpy arrays from SPLITS_DIR."""
    print("[1/4] Loading sequences from disk …")
    data = {}
    for split in ('train', 'val', 'test'):
        X       = np.load(os.path.join(SPLITS_DIR, f'X_{split}.npy'))
        y_fault = np.load(os.path.join(SPLITS_DIR, f'y_fault_{split}.npy'))
        y_label = np.load(os.path.join(SPLITS_DIR, f'y_label_{split}.npy'))
        data[split] = (X, y_fault, y_label)
        print(f"      [{split}] X={X.shape} fault_dist={np.bincount(y_fault).tolist()} "
              f"label_dist={np.bincount(y_label).tolist()}")
    return data


def get_callbacks(model_path: str):
    """Return EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_fault_type_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_fault_type_accuracy',
            mode='max',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    splits = load_splits()
    X_train, y_fault_train, y_label_train = splits['train']
    X_val,   y_fault_val,   y_label_val   = splits['val']

    # ── 2. Compute sample weights (handles Warning class imbalance) ───────────
    print("[2/4] Computing sample weights to handle class imbalance …")
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_fault_train)
    print(f"      weight range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")

    # ── 3. Build model ────────────────────────────────────────────────────────
    print("[3/4] Building CNN-LSTM model …")
    model = build_model()
    model.summary()
    print(f"      Total parameters: {model.count_params():,}")

    # ── 4. Build tf.data.Dataset (Keras 3 compatible with sample weights) ────
    # Keras 3 requires (x, y, sample_weight) as a Dataset tuple for multi-output
    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices((
            X_train,
            {'fault_type': y_fault_train, 'severity_label': y_label_train},
            sample_weights,
        ))
        .shuffle(buffer_size=10000, seed=42)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        tf.data.Dataset
        .from_tensor_slices((
            X_val,
            {'fault_type': y_fault_val, 'severity_label': y_label_val},
        ))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print(f"\n[4/4] Training for up to {EPOCHS} epochs (batch={BATCH_SIZE}) ...")
    print(f"      Early-stopping patience: {PATIENCE} epochs\n")
    t0 = time.time()

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=get_callbacks(MODEL_PATH),
        verbose=1,
    )

    elapsed = time.time() - t0
    print(f"\n✅ Training finished in {elapsed/60:.1f} min")

    # ── Save training history ─────────────────────────────────────────────────
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(HISTORY_PATH, 'w') as f:
        json.dump(hist_dict, f, indent=2)
    print(f"   Training history → {HISTORY_PATH}")

    # ── Quick final metrics ───────────────────────────────────────────────────
    best_epoch = int(np.argmax(history.history['val_fault_type_accuracy'])) + 1
    best_val_acc = max(history.history['val_fault_type_accuracy'])
    print(f"\n   Best val fault_type_accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"   Model saved → {MODEL_PATH}")


if __name__ == '__main__':
    main()
