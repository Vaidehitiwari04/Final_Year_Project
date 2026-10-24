"""
evaluate.py — Full evaluation of the trained CNN-LSTM model on the held-out test set.

Run:
    cd lpg_pipeline_monitor
    python evaluate.py

Outputs (saved to results/):
    evaluation_report.json  — all metrics as JSON
    confusion_fault.csv     — confusion matrix (fault_type)
    confusion_label.csv     — confusion matrix (severity_label)
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score,
)

from config import (
    SPLITS_DIR, MODELS_DIR, RESULTS_DIR,
    MODEL_PATH, SCALER_PATH, REPORT_PATH,
    FAULT_TYPE_LABELS, FAULT_TYPE_INV,
    SEVERITY_MAP,
)


def load_test_data():
    X       = np.load(os.path.join(SPLITS_DIR, 'X_test.npy'))
    y_fault = np.load(os.path.join(SPLITS_DIR, 'y_fault_test.npy'))
    y_label = np.load(os.path.join(SPLITS_DIR, 'y_label_test.npy'))
    print(f"Test set: X={X.shape}  fault_dist={np.bincount(y_fault).tolist()}")
    return X, y_fault, y_label


def compute_all_metrics(y_true, y_pred_prob, class_names, task_name):
    """Compute accuracy, weighted F1, per-class F1, AUC-ROC, confusion matrix."""
    y_pred = np.argmax(y_pred_prob, axis=1)
    n_classes = len(class_names)

    acc  = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_true, y_pred, average='macro',    zero_division=0)

    # AUC-ROC (one-vs-rest)
    try:
        auc = roc_auc_score(
            y_true, y_pred_prob,
            multi_class='ovr', average='weighted'
        )
    except ValueError:
        auc = None

    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n── {task_name} ──────────────────────────────────────────────")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Weighted F1   : {f1_w:.4f}")
    print(f"  Macro F1      : {f1_m:.4f}")
    if auc:
        print(f"  AUC-ROC (OvR) : {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("  Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=[f'True_{c}' for c in class_names],
                         columns=[f'Pred_{c}' for c in class_names])
    print(cm_df.to_string())

    return {
        'accuracy':    float(acc),
        'f1_weighted': float(f1_w),
        'f1_macro':    float(f1_m),
        'auc_roc':     float(auc) if auc else None,
        'per_class':   report_dict,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
    }, cm_df


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("[1/3] Loading test data and trained model …")
    X_test, y_fault_test, y_label_test = load_test_data()

    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"      Model loaded from {MODEL_PATH}")

    # ── Predict ───────────────────────────────────────────────────────────────
    print("[2/3] Running inference on test set …")
    pred_fault_prob, pred_label_prob = model.predict(X_test, batch_size=512, verbose=1)
    print(f"      Predictions: fault_prob={pred_fault_prob.shape}, "
          f"label_prob={pred_label_prob.shape}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("[3/3] Computing evaluation metrics …")

    fault_metrics, cm_fault = compute_all_metrics(
        y_fault_test, pred_fault_prob,
        class_names=FAULT_TYPE_LABELS,
        task_name='FAULT TYPE CLASSIFICATION (none / leak / blockage)',
    )

    label_metrics, cm_label = compute_all_metrics(
        y_label_test, pred_label_prob,
        class_names=[SEVERITY_MAP[i] for i in range(3)],
        task_name='SEVERITY LABEL CLASSIFICATION (Normal / Warning / Danger)',
    )

    # ── Save results ──────────────────────────────────────────────────────────
    report = {
        'fault_type':     fault_metrics,
        'severity_label': label_metrics,
    }
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Evaluation report saved → {REPORT_PATH}")

    # Save confusion matrices as CSV
    cm_fault.to_csv(os.path.join(RESULTS_DIR, 'confusion_fault.csv'))
    cm_label.to_csv(os.path.join(RESULTS_DIR, 'confusion_label.csv'))
    print(f"   Confusion matrices saved → results/confusion_*.csv")

    # ── Summary banner ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Fault Type  | Accuracy: {fault_metrics['accuracy']:.4f}  "
          f"F1(W): {fault_metrics['f1_weighted']:.4f}  "
          f"AUC: {fault_metrics['auc_roc']:.4f if fault_metrics['auc_roc'] else 'N/A'}")
    print(f"  Severity    | Accuracy: {label_metrics['accuracy']:.4f}  "
          f"F1(W): {label_metrics['f1_weighted']:.4f}  "
          f"AUC: {label_metrics['auc_roc']:.4f if label_metrics['auc_roc'] else 'N/A'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
