"""
predict_service.py — Production Edge Inference Service for LPG Pipeline Monitoring.
Implements predict_window(df_window) for live SCADA telemetry feeds.
"""

import os
import joblib
import numpy as np
import pandas as pd

from config import SENSOR_COLS, FAULT_TYPE_LABELS, SEVERITY_MAP

# Constants
WINDOW_LEN = 12
SLOPE_WEIGHTS = (np.arange(WINDOW_LEN) - (WINDOW_LEN - 1) / 2.0) / np.sum((np.arange(WINDOW_LEN) - (WINDOW_LEN - 1) / 2.0) ** 2)

_MODELS_CACHE = {}

def load_inference_artifacts(models_dir: str = 'models'):
    """Load and cache serialized models for sub-millisecond edge execution."""
    global _MODELS_CACHE
    if _MODELS_CACHE:
        return _MODELS_CACHE

    best_model_path = os.path.join(models_dir, 'best_model.joblib')
    danger_model_path = os.path.join(models_dir, 'danger_model.joblib')
    residual_model_path = os.path.join(models_dir, 'residual_model.joblib')
    unsupervised_path = os.path.join(models_dir, 'unsupervised_and_thresholds.joblib')

    if not all(os.path.exists(p) for p in [best_model_path, danger_model_path, residual_model_path]):
        raise FileNotFoundError(f"One or more model artifacts missing from {models_dir}!")

    _MODELS_CACHE['clf_fault'] = joblib.load(best_model_path)
    _MODELS_CACHE['clf_danger'] = joblib.load(danger_model_path)
    _MODELS_CACHE['res_reg'] = joblib.load(residual_model_path)
    
    if os.path.exists(unsupervised_path):
        _MODELS_CACHE['thresholds'] = joblib.load(unsupervised_path)
    else:
        _MODELS_CACHE['thresholds'] = {'warn_cutoff': 0.40, 'danger_cutoff': 0.70}

    return _MODELS_CACHE

def extract_single_window_features(df_window: pd.DataFrame, res_reg) -> pd.DataFrame:
    """Extract 39 features from a single 60s (12-row) window."""
    if len(df_window) != WINDOW_LEN:
        raise ValueError(f"Expected window length {WINDOW_LEN} rows, got {len(df_window)} rows!")

    feats = {}
    
    # a) Raw features (mean, std, slope)
    for col in SENSOR_COLS:
        vals = df_window[col].values
        feats[f'{col}_mean'] = float(np.mean(vals))
        feats[f'{col}_std'] = float(np.std(vals))
        feats[f'{col}_slope'] = float(np.dot(vals, SLOPE_WEIGHTS))

    # b) Flow-compensated residuals
    res_df = res_reg.transform(df_window)
    for col in ['res_pressure', 'res_vibration', 'res_ultrasonic']:
        vals = res_df[col].values
        feats[f'{col}_mean'] = float(np.mean(vals))
        feats[f'{col}_std'] = float(np.std(vals))
        feats[f'{col}_slope'] = float(np.dot(vals, SLOPE_WEIGHTS))

    # c) Smoothed delta / lag features
    half = WINDOW_LEN // 2
    for col in SENSOR_COLS:
        vals = df_window[col].values
        half1_mean = np.mean(vals[:half])
        half2_mean = np.mean(vals[half:])
        feats[f'{col}_delta_half'] = float(half2_mean - half1_mean)
        feats[f'{col}_delta_step'] = float(vals[-1] - vals[0])

    return pd.DataFrame([feats])

def predict_window(df_window: pd.DataFrame, models_dir: str = 'models') -> dict:
    """
    Main edge interface function.
    
    Parameters
    ----------
    df_window : pd.DataFrame
        DataFrame of 12 consecutive sensor readings (60s) containing SENSOR_COLS.
        Blank sensor dropouts are filled by forward/backward fill.
        
    Returns
    -------
    dict:
        {
            'level': 'Normal' | 'Warning' | 'Danger',
            'likely_cause': 'Normal' | 'Leak' | 'Blockage',
            'probability': float,
            'danger_probability': float,
            'fault_probabilities': dict
        }
    """
    artifacts = load_inference_artifacts(models_dir)
    clf_fault = artifacts['clf_fault']
    clf_danger = artifacts['clf_danger']
    res_reg = artifacts['res_reg']
    thresholds = artifacts['thresholds']
    warn_cutoff = thresholds.get('warn_cutoff', 0.40)
    danger_cutoff = thresholds.get('danger_cutoff', 0.70)

    # Impute dropouts per window if any
    clean_window = df_window[SENSOR_COLS].ffill().bfill().fillna(0.0)

    # Feature extraction
    X_single = extract_single_window_features(clean_window, res_reg)

    # Supervised fault classification
    fault_idx = int(clf_fault.predict(X_single)[0])
    fault_probs = clf_fault.predict_proba(X_single)[0]
    likely_cause = ['Normal', 'Leak', 'Blockage'][fault_idx]
    cause_prob = float(fault_probs[fault_idx])

    # Danger classification
    danger_prob = float(clf_danger.predict_proba(X_single)[0, 1])
    
    if danger_prob >= danger_cutoff:
        level = 'Danger'
    elif danger_prob >= warn_cutoff or fault_idx != 0:
        level = 'Warning'
    else:
        level = 'Normal'

    return {
        'level': level,
        'likely_cause': likely_cause,
        'probability': cause_prob,
        'danger_probability': danger_prob,
        'fault_probabilities': {
            'Normal': float(fault_probs[0]),
            'Leak': float(fault_probs[1]),
            'Blockage': float(fault_probs[2])
        }
    }
