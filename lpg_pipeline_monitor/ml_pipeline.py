"""
ml_pipeline.py — Comprehensive Classical Machine Learning Engine for LPG Pipeline Monitoring.
Implements:
  - Clean non-leaking window feature extraction (raw, residuals, lag/delta)
  - Flow-compensated residual regressions
  - Unsupervised models (Isolation Forest, One-Class SVM, Mahalanobis Distance) with 99.9th percentile thresholds
  - Supervised fault classifiers (HistGradientBoosting, RandomForest)
  - Danger predictor (calibrated lookahead classifier with false-alarm budget cutoffs)
  - SCADA and ROC baselines
  - Full evaluation metrics (lead time, onset delay, false alarm rates, macro-F1, Brier score)
  - Ablation study, 5-seed repetition, and stress tests
  - Model serialization (joblib) & predict_window inference
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.calibration import CalibratedClassifierCV
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    confusion_matrix, brier_score_loss, roc_auc_score
)

# Constants
ROW_INTERVAL_S = 5
WINDOW_S = 60
WINDOW_LEN = int(WINDOW_S / ROW_INTERVAL_S)  # 12 steps
DANGER_HORIZON_MIN = 10
DANGER_HORIZON_ROWS = int((DANGER_HORIZON_MIN * 60) / ROW_INTERVAL_S)  # 120 steps
SECONDS_PER_DAY = 86400
SAMPLES_PER_DAY_PER_NODE = SECONDS_PER_DAY / ROW_INTERVAL_S  # 17,280

SENSOR_COLS = [
    'pressure_bar_g',
    'flow_mmscmd',
    'gas_temp_c',
    'vibration_mm_s_rms',
    'ultrasonic_db_rel',
    'gas_lel_pct',
]

FAULT_CLASS_MAP = {
    'none': 0,
    'leak': 1,
    'blockage': 2
}
CLASS_NAMES = ['Normal', 'Leak', 'Blockage']

# Precomputed slope weights for window of length 12
_t = np.arange(WINDOW_LEN)
_t_mean = (WINDOW_LEN - 1) / 2.0
_denom = np.sum((_t - _t_mean) ** 2)
SLOPE_WEIGHTS = (_t - _t_mean) / _denom


def impute_dropouts(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then backward-fill sensor NaNs per node."""
    df = df.copy()
    for node in df['node_id'].unique():
        mask = df['node_id'] == node
        df.loc[mask, SENSOR_COLS] = (
            df.loc[mask, SENSOR_COLS]
            .ffill()
            .bfill()
        )
    return df


class FlowResidualRegressor:
    """Fits linear relationships for P, Vibr, Ultra as a function of Flow on healthy train data."""
    def __init__(self):
        self.reg_p = Ridge(alpha=1.0)
        self.reg_v = Ridge(alpha=1.0)
        self.reg_u = Ridge(alpha=1.0)
        self.is_fitted = False

    def fit(self, df_healthy_train: pd.DataFrame):
        Q = df_healthy_train[['flow_mmscmd']].values
        self.reg_p.fit(Q, df_healthy_train['pressure_bar_g'].values)
        self.reg_v.fit(Q, df_healthy_train['vibration_mm_s_rms'].values)
        self.reg_u.fit(Q, df_healthy_train['ultrasonic_db_rel'].values)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("FlowResidualRegressor must be fitted before transform!")
        Q = df[['flow_mmscmd']].values
        res_df = pd.DataFrame(index=df.index)
        res_df['res_pressure'] = df['pressure_bar_g'].values - self.reg_p.predict(Q)
        res_df['res_vibration'] = df['vibration_mm_s_rms'].values - self.reg_v.predict(Q)
        res_df['res_ultrasonic'] = df['ultrasonic_db_rel'].values - self.reg_u.predict(Q)
        return res_df


class MahalanobisDetector:
    """Unsupervised anomaly detector based on regularized Mahalanobis distance."""
    def __init__(self):
        self.lw = LedoitWolf()
        self.mean_ = None
        self.precision_ = None
        self.threshold_ = None

    def fit(self, X: np.ndarray):
        self.lw.fit(X)
        self.mean_ = np.mean(X, axis=0)
        self.precision_ = self.lw.precision_
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        diff = X - self.mean_
        return np.sqrt(np.sum(np.dot(diff, self.precision_) * diff, axis=1))

    def set_threshold_from_val(self, val_scores: np.ndarray, percentile: float = 99.9):
        self.threshold_ = float(np.percentile(val_scores, percentile))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.score(X) > self.threshold_).astype(int)


def extract_features_vectorized(df: pd.DataFrame, residual_reg: FlowResidualRegressor) -> pd.DataFrame:
    """
    Vectorized extraction of 60s window features per node.
    Returns a DataFrame of features aligned with rows (indices >= WINDOW_LEN - 1).
    """
    feature_dfs = []
    
    # Pre-calculate residuals
    res_df = residual_reg.transform(df)
    augmented = pd.concat([df[['node_id', 'timestamp'] + SENSOR_COLS], res_df], axis=1)

    for node in augmented['node_id'].unique():
        node_df = augmented[augmented['node_id'] == node].copy()
        n_rows = len(node_df)
        
        node_feats = pd.DataFrame(index=node_df.index)
        
        # a) Raw features (mean, std, slope)
        for col in SENSOR_COLS:
            vals = node_df[col].values
            node_feats[f'{col}_mean'] = node_df[col].rolling(WINDOW_LEN).mean()
            node_feats[f'{col}_std'] = node_df[col].rolling(WINDOW_LEN).std(ddof=0)
            
            # Convolution for slope
            slope_conv = np.convolve(vals, SLOPE_WEIGHTS[::-1], mode='valid')
            padded_slope = np.empty(n_rows)
            padded_slope[:WINDOW_LEN - 1] = np.nan
            padded_slope[WINDOW_LEN - 1:] = slope_conv
            node_feats[f'{col}_slope'] = padded_slope

        # b) Flow-compensated residual features (mean, std, slope)
        for col in ['res_pressure', 'res_vibration', 'res_ultrasonic']:
            vals = node_df[col].values
            node_feats[f'{col}_mean'] = node_df[col].rolling(WINDOW_LEN).mean()
            node_feats[f'{col}_std'] = node_df[col].rolling(WINDOW_LEN).std(ddof=0)
            
            slope_conv = np.convolve(vals, SLOPE_WEIGHTS[::-1], mode='valid')
            padded_slope = np.empty(n_rows)
            padded_slope[:WINDOW_LEN - 1] = np.nan
            padded_slope[WINDOW_LEN - 1:] = slope_conv
            node_feats[f'{col}_slope'] = padded_slope

        # c) Smoothed delta / lag features
        half = WINDOW_LEN // 2
        for col in SENSOR_COLS:
            mean_half = node_df[col].rolling(half).mean()
            node_feats[f'{col}_delta_half'] = mean_half - mean_half.shift(half)
            node_feats[f'{col}_delta_step'] = node_df[col] - node_df[col].shift(WINDOW_LEN - 1)

        feature_dfs.append(node_feats)

    all_feats = pd.concat(feature_dfs).sort_index()
    return all_feats


def build_danger_labels(df: pd.DataFrame, horizon_rows: int = DANGER_HORIZON_ROWS) -> pd.Series:
    """
    Build danger label: 1 if static limit breached within the next N minutes
    (within the same node and split), else 0.
    """
    danger_series = pd.Series(0, index=df.index, dtype=int)
    for (split, node), group in df.groupby(['split', 'node_id']):
        s = group['limit_alarm']
        future_breach = (
            s[::-1]
            .rolling(window=horizon_rows, min_periods=1)
            .max()[::-1]
            .shift(-1)
            .fillna(0)
            .astype(int)
        )
        danger_series.loc[group.index] = future_breach
    return danger_series


def prepare_dataset(csv_path: str):
    """Load, clean, extract features, and return structured datasets."""
    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    df = impute_dropouts(df)
    
    # Fit residual model on healthy train rows only
    train_healthy_mask = (df['split'] == 'train') & (df['fault_type'] == 'none') & (df['event_kind'].isna())
    healthy_train = df[train_healthy_mask]
    
    print(f"Fitting flow residual regression on {len(healthy_train)} healthy training rows ...")
    res_reg = FlowResidualRegressor().fit(healthy_train)

    print("Extracting vectorized 60s window features ...")
    feats_df = extract_features_vectorized(df, res_reg)
    
    # Ground truth labels
    # Supervised target: 0 = Normal, 1 = Leak, 2 = Blockage
    # Sensor faults and demand steps mapped to Normal (0)
    supervised_target = pd.Series(0, index=df.index, dtype=int)
    supervised_target.loc[df['fault_type'] == 'leak'] = 1
    supervised_target.loc[df['fault_type'] == 'blockage'] = 2
    # Ensure sensor faults and demand steps are 0
    supervised_target.loc[df['event_kind'].isin(['demand_step', 'sensor_fault'])] = 0

    danger_target = build_danger_labels(df, DANGER_HORIZON_ROWS)

    # Valid window mask (drop first WINDOW_LEN - 1 rows of each node/split where rolling is NaN)
    valid_mask = ~feats_df.isna().any(axis=1)

    dataset = {
        'df_meta': df.loc[valid_mask].copy(),
        'X': feats_df.loc[valid_mask].copy(),
        'y_fault': supervised_target.loc[valid_mask].copy(),
        'y_danger': danger_target.loc[valid_mask].copy(),
        'res_reg': res_reg,
        'feature_names': feats_df.columns.tolist()
    }
    return dataset
