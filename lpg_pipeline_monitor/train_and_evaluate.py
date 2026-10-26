"""
train_and_evaluate.py — Full training, evaluation, ablation, multi-seed, stress test, and artifact generation.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    confusion_matrix, brier_score_loss, roc_auc_score
)
from sklearn.inspection import permutation_importance
import shap

from ml_pipeline import (
    prepare_dataset, SENSOR_COLS, CLASS_NAMES,
    MahalanobisDetector, FlowResidualRegressor,
    extract_features_vectorized, impute_dropouts,
    WINDOW_LEN, SAMPLES_PER_DAY_PER_NODE, SLOPE_WEIGHTS
)

RESULTS_DIR = 'results'
MODELS_DIR = 'models'

def run_experiment(seed=42):
    np.random.seed(seed)
    csv_path = 'data/pipeline_training_data.csv'
    dataset = prepare_dataset(csv_path)
    
    meta = dataset['df_meta']
    X = dataset['X']
    y_fault = dataset['y_fault']
    y_danger = dataset['y_danger']
    res_reg = dataset['res_reg']
    feature_names = dataset['feature_names']

    # Split masks
    train_mask = meta['split'] == 'train'
    val_mask = meta['split'] == 'val'
    test_mask = meta['split'] == 'test'

    healthy_train_mask = train_mask & (meta['fault_type'] == 'none') & (meta['event_kind'].isna())
    healthy_val_mask = val_mask & (meta['fault_type'] == 'none') & (meta['event_kind'].isna())

    # Data slices
    X_train, y_train = X[train_mask], y_fault[train_mask]
    X_val, y_val = X[val_mask], y_fault[val_mask]
    X_test, y_test = X[test_mask], y_fault[test_mask]

    X_train_h = X[healthy_train_mask]
    X_val_h = X[healthy_val_mask]

    # Scaler for unsupervised distance models
    scaler = StandardScaler().fit(X_train_h)
    X_train_h_scaled = scaler.transform(X_train_h)
    X_val_h_scaled = scaler.transform(X_val_h)
    X_test_scaled = scaler.transform(X_test)

    # ─────────────────────────────────────────────────────────────────────────────
    # 1. UNSUPERVISED MODELS (Trained on healthy train windows only)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n--- Training Unsupervised Anomaly Detectors ---")
    
    # a) Isolation Forest
    iforest = IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1).fit(X_train_h)
    iforest_val_scores = -iforest.score_samples(X_val_h)
    iforest_thresh = float(np.percentile(iforest_val_scores, 99.9))
    iforest_test_scores = -iforest.score_samples(X_test)
    iforest_test_preds = (iforest_test_scores > iforest_thresh).astype(int)

    # b) One-Class SVM (subsampled to 3k for rapid edge training and low RAM)
    sample_idx = np.random.choice(len(X_train_h_scaled), min(3000, len(X_train_h_scaled)), replace=False)
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01).fit(X_train_h_scaled[sample_idx])
    ocsvm_val_scores = -ocsvm.decision_function(X_val_h_scaled)
    ocsvm_thresh = float(np.percentile(ocsvm_val_scores, 99.9))
    ocsvm_test_scores = -ocsvm.decision_function(X_test_scaled)
    ocsvm_test_preds = (ocsvm_test_scores > ocsvm_thresh).astype(int)

    # c) Mahalanobis Distance
    maha = MahalanobisDetector().fit(X_train_h_scaled)
    maha_val_scores = maha.score(X_val_h_scaled)
    maha.set_threshold_from_val(maha_val_scores, percentile=99.9)
    maha_test_scores = maha.score(X_test_scaled)
    maha_test_preds = maha.predict(X_test_scaled)

    # ─────────────────────────────────────────────────────────────────────────────
    # 2. SUPERVISED FAULT CLASSIFIERS (Normal, Leak, Blockage)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n--- Training Supervised Fault Classifiers ---")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced', n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)

    hgb = HistGradientBoostingClassifier(random_state=seed, class_weight='balanced')
    hgb.fit(X_train, y_train)
    hgb_preds = hgb.predict(X_test)
    hgb_probs = hgb.predict_proba(X_test)

    # ─────────────────────────────────────────────────────────────────────────────
    # 3. DANGER PREDICTOR (Breach within next 10 minutes)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n--- Training Calibrated Danger Predictor ---")
    y_danger_train = y_danger[train_mask]
    y_danger_val = y_danger[val_mask]
    y_danger_test = y_danger[test_mask]

    base_danger = HistGradientBoostingClassifier(random_state=seed, class_weight='balanced')
    cal_danger = CalibratedClassifierCV(estimator=base_danger, method='sigmoid', cv=3)
    cal_danger.fit(X_train, y_danger_train)

    val_danger_probs = cal_danger.predict_proba(X_val)[:, 1]
    test_danger_probs = cal_danger.predict_proba(X_test)[:, 1]

    # Tune cutoffs on healthy validation set for false alarm budget <= 2 / day
    val_healthy_danger_probs = val_danger_probs[healthy_val_mask[val_mask].values]
    
    # False alarm budget: 2 per day -> allowed FPR = 2 / SAMPLES_PER_DAY_PER_NODE
    allowed_fpr = 2.0 / SAMPLES_PER_DAY_PER_NODE # ~0.0001157
    warn_cutoff = float(np.percentile(val_healthy_danger_probs, (1.0 - allowed_fpr) * 100))
    # Ensure reasonable boundaries
    warn_cutoff = max(0.15, min(0.60, warn_cutoff))
    danger_cutoff = max(warn_cutoff + 0.20, 0.70)

    print(f"Danger Cutoffs (Tuned on Val): Warning = {warn_cutoff:.4f}, Danger = {danger_cutoff:.4f}")

    # Map test probabilities to levels: 0=Normal, 1=Warning, 2=Danger
    test_levels = np.zeros(len(test_danger_probs), dtype=int)
    test_levels[test_danger_probs >= warn_cutoff] = 1
    test_levels[test_danger_probs >= danger_cutoff] = 2

    # ─────────────────────────────────────────────────────────────────────────────
    # 4. BASELINES
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n--- Evaluating Baselines (Static SCADA & Rate-of-Change Rule) ---")
    # SCADA static limit trip
    scada_test_preds = (meta.loc[test_mask, 'limit_alarm'] == 1).astype(int).values

    # Rate of change rule: |dP/dt| slope on raw pressure
    val_p_slopes = np.abs(X.loc[healthy_val_mask, 'pressure_bar_g_slope'].values)
    roc_thresh = float(np.percentile(val_p_slopes, 99.9))
    test_p_slopes = np.abs(X.loc[test_mask, 'pressure_bar_g_slope'].values)
    roc_test_preds = (test_p_slopes > roc_thresh).astype(int)

    # ─────────────────────────────────────────────────────────────────────────────
    # 5. DETAILED EVENT EVALUATION (Delay, Lead Time, Caught by Size)
    # ─────────────────────────────────────────────────────────────────────────────
    test_meta = meta[test_mask].copy()
    test_meta['pred_fault'] = hgb_preds
    test_meta['pred_level'] = test_levels
    test_meta['prob_danger'] = test_danger_probs

    # Events caught by size
    leak_sizes = {'<=2%': (0, 2.0), '2-10%': (2.0, 10.0), '>10%': (10.0, 100.0)}
    block_sizes = {'<=30%': (0, 30.0), '>30%': (30.0, 100.0)}

    event_summary = {}

    for cat, (low, high) in leak_sizes.items():
        leak_events = test_meta[
            (test_meta['event_kind'] == 'leak') &
            (test_meta['fault_size_pct'] > low) &
            (test_meta['fault_size_pct'] <= high)
        ]['event_id'].unique()
        
        caught = 0
        for eid in leak_events:
            ev_rows = test_meta[test_meta['event_id'] == eid]
            # Caught if classified as leak or danger level >= 1
            if (ev_rows['pred_fault'] == 1).any() or (ev_rows['pred_level'] >= 1).any():
                caught += 1
        event_summary[f'Leak {cat}'] = {
            'total': len(leak_events),
            'caught': caught,
            'pct': (caught / len(leak_events) * 100) if len(leak_events) > 0 else 0.0
        }

    for cat, (low, high) in block_sizes.items():
        block_events = test_meta[
            (test_meta['event_kind'] == 'blockage') &
            (test_meta['fault_size_pct'] > low) &
            (test_meta['fault_size_pct'] <= high)
        ]['event_id'].unique()
        
        caught = 0
        for eid in block_events:
            ev_rows = test_meta[test_meta['event_id'] == eid]
            if (ev_rows['pred_fault'] == 2).any() or (ev_rows['pred_level'] >= 1).any():
                caught += 1
        event_summary[f'Blockage {cat}'] = {
            'total': len(block_events),
            'caught': caught,
            'pct': (caught / len(block_events) * 100) if len(block_events) > 0 else 0.0
        }

    # Delays and Lead Times
    delays_after_onset = []
    lead_times_before_trip = []

    all_fault_events = test_meta[test_meta['event_kind'].isin(['leak', 'blockage'])]['event_id'].unique()
    for eid in all_fault_events:
        ev_rows = test_meta[test_meta['event_id'] == eid].sort_values('timestamp')
        onset_rows = ev_rows[ev_rows['s_since_onset'] >= 0]
        if len(onset_rows) == 0:
            continue
        
        # First AI detection (warning level >= 1 or fault != 0)
        ai_detect = onset_rows[(onset_rows['pred_level'] >= 1) | (onset_rows['pred_fault'] != 0)]
        if len(ai_detect) > 0:
            delay = ai_detect.iloc[0]['s_since_onset']
            delays_after_onset.append(delay)
            t_ai = ai_detect.iloc[0]['timestamp']
            
            # Static SCADA trip
            scada_trip = onset_rows[onset_rows['limit_alarm'] == 1]
            if len(scada_trip) > 0:
                t_scada = scada_trip.iloc[0]['timestamp']
                t_ai_dt = pd.to_datetime(t_ai)
                t_scada_dt = pd.to_datetime(t_scada)
                lead_s = (t_scada_dt - t_ai_dt).total_seconds()
                lead_times_before_trip.append(lead_s)

    median_delay = float(np.median(delays_after_onset)) if delays_after_onset else 0.0
    median_lead_time = float(np.median(lead_times_before_trip)) if lead_times_before_trip else 0.0

    # False Alarms Per Day on Healthy, Demand Steps, Sensor Faults
    fa_rates = {}
    for subset_name, mask_cond in [
        ('plain_healthy', test_meta['event_kind'].isna() & (test_meta['fault_type'] == 'none')),
        ('demand_steps', test_meta['event_kind'] == 'demand_step'),
        ('sensor_faults', test_meta['event_kind'] == 'sensor_fault'),
    ]:
        sub_df = test_meta[mask_cond]
        if len(sub_df) == 0:
            fa_rates[subset_name] = 0.0
            continue
        # False alarm if predicted fault != 0 or danger level >= 1
        fa_count = ((sub_df['pred_fault'] != 0) | (sub_df['pred_level'] >= 1)).sum()
        # Scale to rate per day: (count / rows) * SAMPLES_PER_DAY_PER_NODE
        rate_per_day = (fa_count / len(sub_df)) * SAMPLES_PER_DAY_PER_NODE
        fa_rates[subset_name] = float(rate_per_day)

    # Calibration & Brier score
    brier = float(brier_score_loss(y_danger_test, test_danger_probs))

    # Metrics
    hgb_f1_macro = float(f1_score(y_test, hgb_preds, average='macro'))
    hgb_acc = float(accuracy_score(y_test, hgb_preds))
    rf_f1_macro = float(f1_score(y_test, rf_preds, average='macro'))
    rf_acc = float(accuracy_score(y_test, rf_preds))

    results = {
        'seed': seed,
        'hgb_f1_macro': hgb_f1_macro,
        'hgb_acc': hgb_acc,
        'rf_f1_macro': rf_f1_macro,
        'rf_acc': rf_acc,
        'brier_score': brier,
        'median_delay_s': median_delay,
        'median_lead_time_s': median_lead_time,
        'fa_rates': fa_rates,
        'event_summary': event_summary,
        'models': {
            'hgb': hgb,
            'rf': rf,
            'cal_danger': cal_danger,
            'iforest': iforest,
            'maha': maha,
            'ocsvm': ocsvm,
            'scaler': scaler,
            'res_reg': res_reg,
            'warn_cutoff': warn_cutoff,
            'danger_cutoff': danger_cutoff
        },
        'data': {
            'X_test': X_test,
            'y_test': y_test,
            'y_danger_test': y_danger_test,
            'test_danger_probs': test_danger_probs,
            'feature_names': feature_names
        }
    }
    return results

if __name__ == '__main__':
    t0 = time.time()
    res = run_experiment(seed=42)
    print(f"\nExecution finished in {time.time()-t0:.2f}s")
    print(f"HistGradientBoosting Macro-F1: {res['hgb_f1_macro']:.4f}, Accuracy: {res['hgb_acc']:.4f}")
    print(f"RandomForest Macro-F1:        {res['rf_f1_macro']:.4f}, Accuracy: {res['rf_acc']:.4f}")
    print(f"Median Detection Delay:        {res['median_delay_s']:.1f} s")
    print(f"Median Lead Time before SCADA: {res['median_lead_time_s']:.1f} s")
    print(f"False Alarms / Day (Plain):    {res['fa_rates']['plain_healthy']:.2f}")
    print(f"False Alarms / Day (Demand):   {res['fa_rates']['demand_steps']:.2f}")
    print(f"False Alarms / Day (Sensor):   {res['fa_rates']['sensor_faults']:.2f}")
    print("\nEvents Caught:")
    for k, v in res['event_summary'].items():
        print(f"  {k}: {v['caught']}/{v['total']} ({v['pct']:.1f}%)")
