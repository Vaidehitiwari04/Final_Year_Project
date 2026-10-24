# LPG Pipeline Monitoring System

> **CNN-LSTM Real-Time Fault Detection & Severity Classification**  
> Final Year Project — Deep Learning for Industrial IoT Safety

---

## Project Overview

This system monitors an LPG (Liquefied Petroleum Gas) transmission pipeline across three monitoring stations (`GT-01`, `GT-02`, `GT-03`) and classifies sensor telemetry in real-time using a CNN-LSTM deep learning model.

**Two simultaneous classification targets:**
| Task | Classes | Purpose |
|---|---|---|
| `fault_type` | none / leak / blockage | Identify fault category |
| `severity_label` | Normal / Warning / Danger | Grade fault severity |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline (preprocess + train + evaluate)
cd lpg_pipeline_monitor
python run_pipeline.py

# 3. Launch monitoring dashboard
streamlit run dashboard.py
```

---

## Directory Structure

```
FYP/
├── pipeline_training_data.csv          Raw dataset (21.9 MB, 207,360 rows)
├── deep_analysis.py                    Dataset profiling script
├── dataset_analysis_raw.json           Machine-readable analysis output
└── lpg_pipeline_monitor/
    ├── config.py                       Central configuration
    ├── preprocess.py                   Data imputation, scaling, sequence generation
    ├── model.py                        CNN-LSTM dual-head model definition
    ├── train.py                        Training pipeline with class weights
    ├── evaluate.py                     Full evaluation (F1, AUC-ROC, confusion)
    ├── dashboard.py                    Streamlit live monitoring dashboard
    ├── run_pipeline.py                 One-click: preprocess → train → evaluate
    ├── requirements.txt                Python dependencies
    ├── data/
    │   ├── pipeline_training_data.csv  Dataset (copy inside project)
    │   └── splits/                     Processed numpy arrays (after preprocess.py)
    │       ├── X_train.npy / X_val.npy / X_test.npy
    │       ├── y_fault_train.npy / ...
    │       └── y_label_train.npy / ...
    ├── models/
    │   ├── best_model.keras            Trained model checkpoint (after train.py)
    │   └── scaler.pkl                  Fitted StandardScaler (after preprocess.py)
    └── results/
        ├── training_history.json       Epoch-by-epoch metrics (after train.py)
        ├── evaluation_report.json      Full test metrics (after evaluate.py)
        ├── confusion_fault.csv         Confusion matrix: fault_type
        └── confusion_label.csv         Confusion matrix: severity_label
```

---

## Model Architecture

```
Input: (batch, T=10 steps × 5s = 50-second window, 6 sensor features)
         │
         ▼
Causal Conv1D(64)  → BatchNorm  →  Causal Conv1D(128)  → BatchNorm
         → Causal Conv1D(64)    → BatchNorm  → Dropout(0.3)
         │
         ▼
Bidirectional LSTM(64) → Bidirectional LSTM(32) → Dropout(0.3)
         │
         ▼
Dense(128) → Dropout(0.25) → Dense(64)
         │
    ┌────┴─────┐
    ▼           ▼
Dense(32)    Dense(32)
    │             │
    ▼             ▼
Softmax(3)   Softmax(3)
fault_type   severity_label
```

**Total parameters: ~179,782 (~700 KB)**

### Why CNN-LSTM?
| Component | Role |
|---|---|
| **Causal Conv1D** | Extracts local temporal patterns without future leakage — detects pressure wave steepness (dP/dt), acoustic burst onset |
| **Bidirectional LSTM** | Models long-term sequential trajectory — distinguishes transient demand steps (stabilise) from progressive faults (persist/escalate) |
| **Dual Output Head** | Simultaneous fault identification (what) + severity staging (how bad) |

---

## Dataset

| Property | Value |
|---|---|
| Total Records | 207,360 |
| Monitoring Nodes | GT-01, GT-02, GT-03 |
| Duration | 96 hours (2026-01-01 – 2026-01-04) |
| Sampling Interval | 5 seconds (uniform) |
| Features | 16 columns |
| Duplicates | 0 |
| Event Overlap | 0 (leakage-safe splits) |

### Sensor Features
| Feature | Physical Meaning | Unit |
|---|---|---|
| `pressure_bar_g` | In-line static gauge pressure | bar(g) |
| `flow_mmscmd` | Gas volumetric flow rate | MMSCMD |
| `gas_temp_c` | Fluid temperature (Joule-Thomson effect) | °C |
| `vibration_mm_s_rms` | Mechanical pipe vibration | mm/s RMS |
| `ultrasonic_db_rel` | High-frequency acoustic emission | dB(rel) |
| `gas_lel_pct` | Atmospheric gas concentration | % LEL |

### Class Distribution
| Class | Count | % |
|---|---|---|
| Normal (none/label 0) | 105,071 | 50.67% |
| Leak | 58,915 | 28.41% |
| Blockage | 43,374 | 20.92% |
| Warning (label 1) | 19,618 | 9.46% |
| Danger (label 2) | 82,671 | 39.87% |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Sequence length | 10 steps (50 seconds) |
| Batch size | 256 |
| Optimizer | Adam (lr=1e-3) |
| LR schedule | ReduceLROnPlateau (factor=0.5) |
| Early stopping | patience=8 (val_fault_type_accuracy) |
| Max epochs | 60 |
| Class imbalance | `compute_sample_weight('balanced')` on fault labels |
| Loss weights | fault_type: 1.0, severity_label: 0.4 |

---

## Dashboard Features

The Streamlit dashboard (`dashboard.py`) provides:

- **🔴 Live Monitor**: Replay test-set telemetry with real-time CNN-LSTM inference, node-by-node sensor charts, prediction probability bars, and colour-coded alert banners
- **📊 Model Results**: Training curves, confusion matrices, per-class precision/recall/F1
- **📈 Data Analysis**: Dataset statistics, class distributions, fault signature comparison
- **ℹ️ About**: Architecture overview, pipeline diagram, sensor descriptions
