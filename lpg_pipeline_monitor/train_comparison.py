"""
train_comparison.py — Full ablation study: 3 architectures × 3 feature sets.

Trains and evaluates every combination:
  Models       : CNN-LSTM v2 | Temporal Transformer | TCN
  Feature sets : raw (6F)   | raw+delta (12F)      | raw+residual (7F)

All runs share identical:
  • Random seeds (42)
  • Batch size, max epochs, early-stopping patience, LR schedule
  • compute_sample_weight('balanced') on fault labels

Results saved to  results/comparison/:
  {model}_{feat}_history.json   ← epoch-by-epoch metrics
  {model}_{feat}_eval.json      ← full test-set metrics
  comparison_summary.json       ← all results machine-readable
  comparison_table.md           ← human-readable Markdown report

After all runs the best model checkpoint is copied to
  models/best_model.keras  (overwrites the previous v1 model).

Usage:
    cd lpg_pipeline_monitor
    python train_comparison.py
"""

import os
import json
import shutil
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

from config import (
    SPLITS_DIR, MODELS_DIR, RESULTS_DIR,
    BATCH_SIZE, EPOCHS, PATIENCE, LR_PATIENCE,
)
from model             import build_model              as build_cnn_lstm
from model_transformer import build_transformer_model
from model_tcn         import build_tcn_model
from feature_engineering import build_feature_sets, load_raw_splits

# ── Reproducibility ───────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
COMPARISON_DIR = os.path.join(RESULTS_DIR, 'comparison')

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    'cnn_lstm':    {'builder': build_cnn_lstm,          'label': 'CNN-LSTM v2'},
    'transformer': {'builder': build_transformer_model, 'label': 'Temporal Transformer'},
    'tcn':         {'builder': build_tcn_model,         'label': 'TCN (Dilated Residual)'},
}

FEATURE_SETS = ['raw', 'raw_delta', 'raw_residual']


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _make_datasets(X_tr, y_fault_tr, y_label_tr, X_vl, y_fault_vl, y_label_vl):
    """Wrap numpy arrays in tf.data pipelines with sample weighting."""
    sw = compute_sample_weight('balanced', y_fault_tr)

    train_ds = (
        tf.data.Dataset
        .from_tensor_slices((
            X_tr,
            {'fault_type': y_fault_tr, 'severity_label': y_label_tr},
            sw,
        ))
        .shuffle(buffer_size=10_000, seed=42)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset
        .from_tensor_slices((
            X_vl,
            {'fault_type': y_fault_vl, 'severity_label': y_label_vl},
        ))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds


def _get_callbacks(ckpt_path: str, patience: int):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor='val_fault_type_accuracy',
            mode='max', save_best_only=True, verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_fault_type_accuracy',
            mode='max', patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=LR_PATIENCE, min_lr=1e-6, verbose=0,
        ),
    ]


# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate(model, X_te, y_fault_te, y_label_te) -> dict:
    """Compute classification report + AUC-ROC for both heads."""
    p_fault, p_label = model.predict(X_te, batch_size=512, verbose=0)

    y_fault_pred = np.argmax(p_fault, axis=1)
    y_label_pred = np.argmax(p_label, axis=1)

    n_fault = p_fault.shape[1]
    n_label = p_label.shape[1]

    fault_rep = classification_report(
        y_fault_te, y_fault_pred,
        target_names=['Normal', 'Leak', 'Blockage'],
        output_dict=True,
    )
    label_rep = classification_report(
        y_label_te, y_label_pred,
        target_names=['Normal', 'Warning', 'Danger'],
        output_dict=True,
    )

    fault_auc = roc_auc_score(
        label_binarize(y_fault_te, classes=list(range(n_fault))),
        p_fault, multi_class='ovr', average='macro',
    )
    label_auc = roc_auc_score(
        label_binarize(y_label_te, classes=list(range(n_label))),
        p_label, multi_class='ovr', average='macro',
    )

    return {
        'fault_type': {
            'accuracy':    fault_rep['accuracy'],
            'f1_macro':    fault_rep['macro avg']['f1-score'],
            'f1_weighted': fault_rep['weighted avg']['f1-score'],
            'auc_roc':     fault_auc,
            'per_class':   fault_rep,
        },
        'severity_label': {
            'accuracy':    label_rep['accuracy'],
            'f1_macro':    label_rep['macro avg']['f1-score'],
            'f1_weighted': label_rep['weighted avg']['f1-score'],
            'auc_roc':     label_auc,
            'per_class':   label_rep,
        },
    }


# ── Single training run ───────────────────────────────────────────────────────

def train_one(
    model_key: str, feat_key: str,
    model_cfg: dict, X_dict: dict, base_splits: dict,
) -> dict:
    """Train, evaluate, and persist results for one (model × feature) pair."""
    _, y_fault_tr, y_label_tr = base_splits['train']
    _, y_fault_vl, y_label_vl = base_splits['val']
    _, y_fault_te, y_label_te = base_splits['test']

    X_tr = X_dict['train']
    X_vl = X_dict['val']
    X_te = X_dict['test']
    n_features = X_tr.shape[-1]

    tag = f'{model_key}_{feat_key}'
    print(f"\n{'='*65}")
    print(f"  [{tag}]  {model_cfg['label']}  |  {feat_key} ({n_features}F)")
    print(f"{'='*65}")

    # Check if already done (resume support)
    eval_path = os.path.join(COMPARISON_DIR, f'{tag}_eval.json')
    if os.path.exists(eval_path):
        print(f"  ⏭  Already done — loading saved results.")
        with open(eval_path) as f:
            return json.load(f)

    # Build model
    model = model_cfg['builder'](n_features=n_features)
    print(f"  Parameters: {model.count_params():,}")

    # Datasets
    train_ds, val_ds = _make_datasets(
        X_tr, y_fault_tr, y_label_tr,
        X_vl, y_fault_vl, y_label_vl,
    )

    # Callbacks
    ckpt_path = os.path.join(MODELS_DIR, f'{tag}.keras')
    cbs       = _get_callbacks(ckpt_path, PATIENCE)

    # Train
    t0 = time.time()
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=cbs,
        verbose=1,
    )
    elapsed = time.time() - t0

    best_epoch    = int(np.argmax(history.history['val_fault_type_accuracy'])) + 1
    best_val_acc  = float(max(history.history['val_fault_type_accuracy']))
    print(f"\n  ✅ Best val_fault_acc = {best_val_acc:.4f}  (epoch {best_epoch})")
    print(f"  ⏱  Training time: {elapsed/60:.1f} min")

    # Save history
    hist_path = os.path.join(COMPARISON_DIR, f'{tag}_history.json')
    hist_dict = {k: [float(v) for v in vs]
                 for k, vs in history.history.items()}
    with open(hist_path, 'w') as f:
        json.dump(hist_dict, f, indent=2)

    # Evaluate on test set
    print("  Evaluating on test set …")
    eval_result = _evaluate(model, X_te, y_fault_te, y_label_te)
    eval_result.update({
        'model':                  model_cfg['label'],
        'model_key':              model_key,
        'feature_set':            feat_key,
        'n_features':             n_features,
        'n_params':               model.count_params(),
        'training_time_min':      round(elapsed / 60, 2),
        'best_val_fault_acc':     best_val_acc,
        'best_epoch':             best_epoch,
        'checkpoint':             ckpt_path,
    })

    with open(eval_path, 'w') as f:
        json.dump(eval_result, f, indent=2)

    ft = eval_result['fault_type']
    sl = eval_result['severity_label']
    print(f"  Test fault  → acc={ft['accuracy']:.4f}  F1={ft['f1_macro']:.4f}  AUC={ft['auc_roc']:.4f}")
    print(f"  Test sev    → acc={sl['accuracy']:.4f}  F1={sl['f1_macro']:.4f}  AUC={sl['auc_roc']:.4f}")

    # Free GPU memory between runs
    tf.keras.backend.clear_session()

    return eval_result


# ── Report generation ─────────────────────────────────────────────────────────

def _bar(v: float, total: float = 1.0, width: int = 20) -> str:
    filled = int(round(v / total * width))
    return '█' * filled + '░' * (width - filled)


def generate_report(all_results: list) -> str:
    """Write comparison_table.md and comparison_summary.json."""
    sorted_fault = sorted(all_results,
                          key=lambda r: r['fault_type']['auc_roc'],
                          reverse=True)
    sorted_sev   = sorted(all_results,
                          key=lambda r: r['severity_label']['auc_roc'],
                          reverse=True)

    lines = [
        "# Model Ablation Comparison Report",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d  %H:%M')}  "
        f"| Runs: {len(all_results)}",
        "",
        "---",
        "",
        "## Fault Type Classification — Ranked by AUC-ROC",
        "",
        "| Rank | Model | Feature Set | n_feat | Accuracy | Macro F1 | AUC-ROC | Train min |",
        "|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for rank, r in enumerate(sorted_fault, 1):
        ft = r['fault_type']
        lines.append(
            f"| {rank} | {r['model']} | `{r['feature_set']}` | {r['n_features']} "
            f"| {ft['accuracy']:.4f} | {ft['f1_macro']:.4f} "
            f"| **{ft['auc_roc']:.4f}** | {r['training_time_min']:.1f} |"
        )

    lines += [
        "",
        "## Severity Label Classification — Ranked by AUC-ROC",
        "",
        "| Rank | Model | Feature Set | n_feat | Accuracy | Macro F1 | AUC-ROC |",
        "|:---:|:---|:---|:---:|:---:|:---:|:---:|",
    ]
    for rank, r in enumerate(sorted_sev, 1):
        sl = r['severity_label']
        lines.append(
            f"| {rank} | {r['model']} | `{r['feature_set']}` | {r['n_features']} "
            f"| {sl['accuracy']:.4f} | {sl['f1_macro']:.4f} "
            f"| **{sl['auc_roc']:.4f}** |"
        )

    # Best overall (highest fault AUC)
    best = sorted_fault[0]
    ft_b = best['fault_type']
    sl_b = best['severity_label']
    lines += [
        "",
        "---",
        "",
        "## 🏆 Winner",
        "",
        f"**{best['model']}**  +  **`{best['feature_set']}`** features "
        f"({best['n_features']}F)",
        "",
        f"| Metric | Fault Type | Severity |",
        f"|:---|:---:|:---:|",
        f"| Accuracy      | {ft_b['accuracy']:.4f} | {sl_b['accuracy']:.4f} |",
        f"| Macro F1      | {ft_b['f1_macro']:.4f} | {sl_b['f1_macro']:.4f} |",
        f"| AUC-ROC       | {ft_b['auc_roc']:.4f}  | {sl_b['auc_roc']:.4f}  |",
        f"| Training time | {best['training_time_min']:.1f} min | — |",
        f"| Parameters    | {best['n_params']:,} | — |",
        "",
        f"> Checkpoint: `{best['checkpoint']}`",
        "",
        "---",
        "",
        "## Feature Set Ablation (fault AUC-ROC, averaged across models)",
        "",
    ]

    # Feature set summary
    for feat in FEATURE_SETS:
        rows = [r for r in all_results if r['feature_set'] == feat]
        if rows:
            avg_auc = np.mean([r['fault_type']['auc_roc'] for r in rows])
            lines.append(f"- `{feat}`:  avg AUC-ROC = **{avg_auc:.4f}**  "
                         f"{_bar(avg_auc)}")

    lines += ["", "## Model Ablation (fault AUC-ROC, averaged across feature sets)", ""]
    for mk in MODELS:
        rows = [r for r in all_results if r['model_key'] == mk]
        if rows:
            avg_auc = np.mean([r['fault_type']['auc_roc'] for r in rows])
            lbl = MODELS[mk]['label']
            lines.append(f"- **{lbl}**:  avg AUC-ROC = **{avg_auc:.4f}**  "
                         f"{_bar(avg_auc)}")

    md_path = os.path.join(COMPARISON_DIR, 'comparison_table.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    summary = {
        'generated_at': datetime.now().isoformat(),
        'n_runs': len(all_results),
        'best': {
            'model':        best['model'],
            'model_key':    best['model_key'],
            'feature_set':  best['feature_set'],
            'fault_auc':    ft_b['auc_roc'],
            'fault_acc':    ft_b['accuracy'],
            'sev_auc':      sl_b['auc_roc'],
            'checkpoint':   best['checkpoint'],
        },
        'all_results': all_results,
    }
    json_path = os.path.join(COMPARISON_DIR, 'comparison_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return md_path, best


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    n_total = len(MODELS) * len(FEATURE_SETS)
    print("\n🔥 LPG Pipeline — Multi-Model Ablation Study")
    print(f"   Models:       {list(MODELS.keys())}")
    print(f"   Feature sets: {FEATURE_SETS}")
    print(f"   Total runs:   {n_total}")
    print()

    os.makedirs(COMPARISON_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,     exist_ok=True)

    print("Loading base splits …")
    base_splits = load_raw_splits()

    print("Computing feature sets …")
    feature_Xs = build_feature_sets(base_splits)
    for name, xd in feature_Xs.items():
        nf = list(xd.values())[0].shape[-1]
        print(f"  {name:<15} → {nf} features")

    all_results = []
    run_idx = 0

    for feat_key in FEATURE_SETS:
        X_dict = feature_Xs[feat_key]
        for model_key, model_cfg in MODELS.items():
            run_idx += 1
            print(f"\n── Run {run_idx}/{n_total} ──")
            result = train_one(
                model_key, feat_key, model_cfg, X_dict, base_splits
            )
            all_results.append(result)

    # Generate comparison report
    print("\n\nGenerating comparison report …")
    md_path, best = generate_report(all_results)

    # Copy winner to canonical best_model.keras
    best_ckpt = best['checkpoint']
    canonical = os.path.join(MODELS_DIR, 'best_model.keras')
    if os.path.exists(best_ckpt) and best_ckpt != canonical:
        shutil.copy2(best_ckpt, canonical)
        print(f"  ✅ Best model copied → {canonical}")

    print(f"\n{'='*65}")
    print(f"  🎉 ABLATION STUDY COMPLETE")
    print(f"  🏆 Winner: {best['model']}  +  {best['feature_set']}")
    print(f"     Fault AUC-ROC : {best['fault_type']['auc_roc']:.4f}")
    print(f"     Fault Accuracy: {best['fault_type']['accuracy']:.4f}")
    print(f"  📄 Report  → {md_path}")
    print(f"  📁 Results → {COMPARISON_DIR}")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
