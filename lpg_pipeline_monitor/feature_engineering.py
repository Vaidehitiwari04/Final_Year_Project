"""
feature_engineering.py — Derives enriched feature sets from raw sensor sequences.

Ablation variants produced
  'raw'          (N, T,  6)  original 6 sensor channels
  'raw_delta'    (N, T, 12)  raw + first-order per-step differences
  'raw_residual' (N, T,  7)  raw + flow-compensated pressure residual
                              (fitted on healthy training data only)

Usage (standalone):
    cd lpg_pipeline_monitor
    python feature_engineering.py
"""

import os
import numpy as np
from sklearn.linear_model import Ridge

from config import SPLITS_DIR, N_FEATURES


# ── Feature transforms ────────────────────────────────────────────────────────

def add_delta(X: np.ndarray) -> np.ndarray:
    """Append first-order differences along the time axis as new channels.

    For each timestep t: delta[t] = X[t] - X[t-1]  (delta[0] = 0)

    Parameters
    ----------
    X : np.ndarray  shape (N, T, F)

    Returns
    -------
    np.ndarray  shape (N, T, 2F)
    """
    delta = np.zeros_like(X)
    delta[:, 1:, :] = np.diff(X, axis=1)          # (N, T-1, F) differences
    return np.concatenate([X, delta], axis=-1)      # (N, T, 2F)


def compute_residual_feature(
    X_splits: list,
    y_fault_train: np.ndarray,
) -> tuple:
    """Fit a Ridge regression on healthy-only training timesteps, then append
    the pressure residual (actual − predicted) as a 7th feature channel.

    The model is fitted on normal (fault_type == 0) training sequences ONLY,
    using normalised statistics from that healthy subset, so no information
    from fault events leaks into the feature computation.

    Parameters
    ----------
    X_splits       : list of [X_train, X_val, X_test]  each (N, T, F)
    y_fault_train  : np.ndarray  (N_train,) integer fault labels

    Returns
    -------
    enriched_splits : list of (N, T, F+1) arrays
    reg             : fitted Ridge model (for inspection / serialisation)
    """
    X_train, X_val, X_test = X_splits

    # Extract healthy training sequences only
    healthy_mask = (y_fault_train == 0)
    X_healthy_flat = X_train[healthy_mask].reshape(-1, N_FEATURES)   # (M*T, F)

    print(f"  Fitting residual model on {X_healthy_flat.shape[0]:,} "
          f"healthy timesteps ({healthy_mask.sum()} sequences) …")

    # pressure_bar_g (col 0) ~ all other sensor readings
    reg = Ridge(alpha=1.0)
    reg.fit(X_healthy_flat[:, 1:], X_healthy_flat[:, 0])

    enriched = []
    for X in [X_train, X_val, X_test]:
        N, T, F = X.shape
        X_flat  = X.reshape(N * T, F)
        pred    = reg.predict(X_flat[:, 1:]).reshape(N * T, 1)
        resid   = (X_flat[:, 0:1] - pred).reshape(N, T, 1)
        enriched.append(np.concatenate([X, resid], axis=-1))   # (N, T, F+1)

    return enriched, reg


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_raw_splits(splits_dir: str = SPLITS_DIR) -> dict:
    data = {}
    for split in ('train', 'val', 'test'):
        data[split] = (
            np.load(os.path.join(splits_dir, f'X_{split}.npy')),
            np.load(os.path.join(splits_dir, f'y_fault_{split}.npy')),
            np.load(os.path.join(splits_dir, f'y_label_{split}.npy')),
        )
    return data


def build_feature_sets(base_splits: dict) -> dict:
    """Compute all three feature-set variants from raw splits.

    Returns
    -------
    dict mapping feature-set name → {split_name: X_array}
    """
    X_train, y_fault_train, _ = base_splits['train']
    X_val,   _,             _ = base_splits['val']
    X_test,  _,             _ = base_splits['test']

    sets: dict = {}

    # 1. Raw (6 features) — already exists
    sets['raw'] = {
        'train': X_train,
        'val':   X_val,
        'test':  X_test,
    }

    # 2. Raw + Delta (12 features)
    sets['raw_delta'] = {
        'train': add_delta(X_train),
        'val':   add_delta(X_val),
        'test':  add_delta(X_test),
    }

    # 3. Raw + Residual (7 features)
    res_list, _ = compute_residual_feature(
        [X_train, X_val, X_test], y_fault_train
    )
    sets['raw_residual'] = {
        'train': res_list[0],
        'val':   res_list[1],
        'test':  res_list[2],
    }

    return sets


def save_feature_set(tag: str, X_dict: dict, splits_dir: str = SPLITS_DIR):
    """Save enriched X arrays to disk with a feature-set tag suffix."""
    for split_name, X in X_dict.items():
        path = os.path.join(splits_dir, f'X_{split_name}_{tag}.npy')
        np.save(path, X)
        print(f"  Saved  {path}  shape={X.shape}")


# ── Standalone entry point ────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Feature Engineering — LPG Pipeline Ablation Study")
    print("=" * 65)

    splits = load_raw_splits()
    X_train, _, _ = splits['train']
    print(f"\nRaw splits: train={X_train.shape[0]:,}  "
          f"val={splits['val'][0].shape[0]:,}  "
          f"test={splits['test'][0].shape[0]:,}")
    print(f"Raw features per timestep: {X_train.shape[-1]}\n")

    feat_sets = build_feature_sets(splits)

    for tag, xd in feat_sets.items():
        n_feat = list(xd.values())[0].shape[-1]
        print(f"\n[{tag}]  →  {n_feat} features per timestep")
        save_feature_set(tag, xd)

    print("\n✅ Feature engineering complete!")
    print("   Summary:")
    for tag, xd in feat_sets.items():
        n = list(xd.values())[0].shape[-1]
        print(f"   {tag:<15}  {n:2d} features")


if __name__ == '__main__':
    main()
