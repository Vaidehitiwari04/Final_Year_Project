# -*- coding: utf-8 -*-
"""
preprocess.py — Data loading, imputation, scaling, and sequence generation.

Run:
    cd lpg_pipeline_monitor
    python preprocess.py

Outputs (saved to data/splits/):
    X_train.npy, y_fault_train.npy, y_label_train.npy
    X_val.npy,   y_fault_val.npy,   y_label_val.npy
    X_test.npy,  y_fault_test.npy,  y_label_test.npy
    scaler.pkl  (also copied to models/)
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

from config import (
    SENSOR_COLS, SEQ_LEN, STEP_SIZE,
    FAULT_TYPE_MAP, N_FEATURES,
    DATA_CSV, SPLITS_DIR, SCALER_PATH, MODELS_DIR,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def impute_per_node(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then backward-fill sensor NaNs within each node group."""
    df = df.copy()
    for node in df['node_id'].unique():
        mask = df['node_id'] == node
        df.loc[mask, SENSOR_COLS] = (
            df.loc[mask, SENSOR_COLS]
            .ffill()
            .bfill()
        )
    return df


def make_sequences(df_split: pd.DataFrame, scaler: StandardScaler, seq_len: int = SEQ_LEN):
    """
    Generate sliding-window sequences from a pre-split dataframe.

    Sequences are generated STRICTLY within (node_id × event_group) boundaries:
      - Anomaly rows   → grouped by (node_id, event_id)
      - Baseline rows  → grouped by node_id (contiguous normal baseline)

    Label for each sequence = label of the LAST row in the window.

    Returns
    -------
    X        : ndarray shape (N, seq_len, n_features), dtype float32
    y_fault  : ndarray shape (N,), dtype int32   – fault_type (0/1/2)
    y_label  : ndarray shape (N,), dtype int32   – severity label (0/1/2)
    """
    df = df_split.copy()

    # Scale sensor features
    df[SENSOR_COLS] = scaler.transform(df[SENSOR_COLS]).astype(np.float32)

    # Create group key: node_id || event_id (baseline rows get '__baseline__')
    df['_ek'] = df['event_id'].fillna('__baseline__')
    df['_gk'] = df['node_id'] + '||' + df['_ek']

    X_parts, yf_parts, yl_parts = [], [], []

    for gk, grp in df.groupby('_gk', sort=False):
        grp = grp.sort_values('timestamp')
        vals   = grp[SENSOR_COLS].values.astype(np.float32)
        faults = grp['fault_encoded'].values.astype(np.int32)
        labels = grp['label'].values.astype(np.int32)
        n = len(vals)
        if n < seq_len:
            continue

        # Efficient sliding window via stride tricks
        # Shape: (n - seq_len + 1, seq_len, n_features)
        windows = sliding_window_view(vals, window_shape=(seq_len, N_FEATURES))
        windows = windows[:, 0, :, :]  # squeeze extra dim → (M, seq_len, n_features)

        X_parts.append(windows)
        yf_parts.append(faults[seq_len - 1:])
        yl_parts.append(labels[seq_len - 1:])

    X       = np.concatenate(X_parts, axis=0)
    y_fault = np.concatenate(yf_parts, axis=0)
    y_label = np.concatenate(yl_parts, axis=0)
    return X, y_fault, y_label


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print(f"[1/5] Loading dataset from '{DATA_CSV}' …")
    t0 = time.time()
    df = pd.read_csv(DATA_CSV)
    print(f"      Loaded {len(df):,} rows × {len(df.columns)} cols in {time.time()-t0:.1f}s")

    # ── 2. Impute ─────────────────────────────────────────────────────────────
    print("[2/5] Imputing missing sensor values (ffill -> bfill per node) ...")
    before = df[SENSOR_COLS].isna().sum().sum()
    df = impute_per_node(df)
    after  = df[SENSOR_COLS].isna().sum().sum()
    print(f"      Missing values: {before} -> {after}")

    # ── 3. Encode targets ─────────────────────────────────────────────────────
    print("[3/5] Encoding target columns …")
    df['fault_encoded'] = df['fault_type'].map(FAULT_TYPE_MAP).astype(np.int32)
    # 'label' column is already 0/1/2 int

    # ── 4. Fit scaler on TRAIN only ───────────────────────────────────────────
    print("[4/5] Fitting StandardScaler on training split …")
    train_df = df[df['split'] == 'train']
    val_df   = df[df['split'] == 'val']
    test_df  = df[df['split'] == 'test']

    scaler = StandardScaler()
    scaler.fit(train_df[SENSOR_COLS])

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"      Scaler saved → {SCALER_PATH}")

    # ── 5. Generate sequences ─────────────────────────────────────────────────
    print("[5/5] Generating sliding-window sequences …")

    splits = [
        ('train', train_df),
        ('val',   val_df),
        ('test',  test_df),
    ]

    for name, split_df in splits:
        t1 = time.time()
        X, y_fault, y_label = make_sequences(split_df, scaler, SEQ_LEN)
        np.save(os.path.join(SPLITS_DIR, f'X_{name}.npy'),       X)
        np.save(os.path.join(SPLITS_DIR, f'y_fault_{name}.npy'),  y_fault)
        np.save(os.path.join(SPLITS_DIR, f'y_label_{name}.npy'),  y_label)
        print(f"      [{name}] X={X.shape}  y_fault={y_fault.shape}  "
              f"(fault dist: {np.bincount(y_fault).tolist()})  "
              f"time: {time.time()-t1:.1f}s")

    print("\n[DONE] Preprocessing complete. Sequences saved to:", SPLITS_DIR)


if __name__ == '__main__':
    main()
