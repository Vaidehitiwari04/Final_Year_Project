# Audit Log of Changes: CNN-LSTM Replacement with Classical ML

This document details every file deleted, created, and modified in accordance with the project refactoring objective:
Replacing the CNN-LSTM deep learning architecture with physics-grounded Classical Machine Learning models (Scikit-Learn).

---

## 1. Deleted Files (Deep Learning & CNN-LSTM Code Removal)

The following files were completely removed from the repository:

| File Path | Description & Reason for Deletion |
| :--- | :--- |
| `lpg_pipeline_monitor/model.py` | CNN-LSTM dual-head model definition (Conv1D + BiLSTM + Keras layers). |
| `lpg_pipeline_monitor/model_tcn.py` | Dilated Causal Conv1D Temporal Convolutional Network model in Keras. |
| `lpg_pipeline_monitor/model_transformer.py` | Temporal Transformer multi-head self-attention model in Keras. |
| `lpg_pipeline_monitor/train.py` | TensorFlow/Keras training loop, datasets, and callbacks for CNN-LSTM. |
| `lpg_pipeline_monitor/train_comparison.py` | Deep learning comparison training harness for CNN-LSTM vs. TCN vs. Transformer. |
| `lpg_pipeline_monitor/evaluate.py` | Evaluation script loading `.keras` checkpoints with `tf.keras.models.load_model`. |
| `lpg_pipeline_monitor/models/best_model.keras` | Serialized Keras neural network weights for CNN-LSTM. |
| `lpg_pipeline_monitor/models/cnn_lstm_raw.keras` | Saved checkpoint for raw sensor CNN-LSTM model. |
| `lpg_pipeline_monitor/models/transformer_raw.keras` | Saved checkpoint for raw sensor Transformer model. |
| `lpg_pipeline_monitor/results/comparison/cnn_lstm_raw_eval.json` | JSON test evaluation metrics for old CNN-LSTM. |
| `lpg_pipeline_monitor/results/comparison/cnn_lstm_raw_history.json` | Epoch-by-epoch training logs for old CNN-LSTM. |

---

## 2. Newly Created Files (Classical ML Pipeline & Evaluation)

| File Path | Description & Purpose |
| :--- | :--- |
| `lpg_pipeline_monitor/ml_pipeline.py` | Production classical ML engine: vectorized 60s window feature extraction, flow-compensated residual regression, unsupervised detectors (Isolation Forest, OC-SVM, Mahalanobis), and danger label construction. |
| `lpg_pipeline_monitor/predict_service.py` | Real-time edge inference service implementing `predict_window(df_window)` returning level, likely cause, and probability. |
| `lpg_pipeline_monitor/run_full_evaluation.py` | Comprehensive benchmark runner executing ablation studies, 5-seed repetitions, stress tests (noise $\times 2$, dropouts), SHAP plots, and summary export. |
| `pipeline_ml.ipynb` (and `lpg_pipeline_monitor/pipeline_ml.ipynb`) | Interactive, end-to-end Jupyter Notebook with detailed code cells, visualizations, and markdown commentary. |
| `results/summary.csv` | Machine-readable table of all benchmark metrics (Macro-F1, accuracy, delays, lead times, false alarm rates). |
| `results/summary.md` | Detailed scientific summary report detailing performance across fault sizes and operational conditions. |
| `results/ablation_study.csv` | Empirical feature contribution metrics (Raw vs. Raw+Delta vs. Raw+Residuals vs. Full 39F). |
| `results/multi_seed_metrics.csv` | 5-seed repetition results reporting mean $\pm$ standard deviation. |
| `results/shap_summary.png` | Tree SHAP summary plot illustrating feature attribution for fault isolation. |
| `results/reliability_diagram.png` | Calibration curve verifying probability reliability and Brier score. |
| `models/best_model.joblib` | Saved HistGradientBoostingClassifier model checkpoint. |
| `models/danger_model.joblib` | Saved CalibratedClassifierCV 10-minute danger prediction model. |
| `models/random_forest.joblib` | Saved RandomForestClassifier model checkpoint. |
| `models/residual_model.joblib` | Saved flow-compensated residual linear regression parameters. |
| `models/unsupervised_and_thresholds.joblib` | Saved Isolation Forest, Mahalanobis detector, and calibrated decision cutoffs. |
| `CHANGES.md` | This audit log of repository changes. |

---

## 3. Modified Files

| File Path | Changes Made |
| :--- | :--- |
| `lpg_pipeline_monitor/requirements.txt` | Removed `tensorflow>=2.13`. Added `scipy`, `joblib>=1.3`, `shap>=0.42`, and `matplotlib>=3.7`. |
| `lpg_pipeline_monitor/config.py` | Updated `MODEL_PATH` from `.keras` to `.joblib`, updated report and history paths to point to ML benchmark summaries. |
| `lpg_pipeline_monitor/run_pipeline.py` | Updated one-click runner to invoke `run_full_evaluation.py` instead of deep learning scripts. |
| `lpg_pipeline_monitor/dashboard.py` | Replaced dynamic TensorFlow / Keras loading with Joblib and `predict_service.predict_window()`. Fixed ground-truth parsing bug (`int('none')` ValueError) with robust type casting. Transformed UI from dark theme to modern, high-contrast bright white light theme with standard operational colors (Green/Amber/Red) across all metrics, badges, and Plotly charts. |
| `.streamlit/config.toml` & `lpg_pipeline_monitor/.streamlit/config.toml` | Configured default Streamlit light theme (`base = "light"`, `#ffffff` background, `#0f172a` text). |
| `lpg_pipeline_monitor/README.md` & `README.md` | Re-authored to document the transition to Classical ML, architecture specifications, benchmark comparisons, and edge deployment feasibility. |
| `dataset_analysis_report.md` | Updated Section 6 justification from CNN-LSTM to Classical Machine Learning modeling. |

