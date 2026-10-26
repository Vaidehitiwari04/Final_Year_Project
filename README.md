# Real-Time Monitoring of Oil and Gas Assets Using Edge AI

> **Physics-Informed Classical Machine Learning Fault Detection & Danger Prediction**  
> Final Year Project — Edge AI for Industrial IoT Safety (ESP32 / Wokwi / Microcontroller Pipeline)

---

## 1. Project Overview & Architectural Design

This system monitors an LPG (Liquefied Petroleum Gas) transmission pipeline segment across three monitoring stations (`GT-01`, `GT-02`, `GT-03`).

### Edge AI for Industrial IoT Safety
Traditional SCADA systems rely on static limit alarms (e.g., pressure falling below 45 bar), which only trigger *after* catastrophic rupture has occurred, providing zero early-warning lead time. Rate-of-change rules suffer from frequent false trips caused by electrical noise and normal customer demand shifts.

This project implements a high-performance, physics-informed classical machine learning framework designed specifically for low-power edge microcontrollers (e.g., **ESP32** with 320 KB – 512 KB SRAM):
* **Physics-Grounded Fluid Modeling**: Pipeline fluid flow obeys physical conservation laws (e.g., pressure drop across an orifice $\Delta P \propto Q^2$). Classical tree ensembles with **flow-compensated residuals** directly exploit these relationships to neutralize operational demand noise.
* **Supervised Fault Classifier**: `HistGradientBoostingClassifier` and `RandomForestClassifier` (categorizing Normal vs. Leak vs. Blockage; transient demand steps and sensor faults mapped to Normal).
* **Unsupervised Anomaly Detectors**: Isolation Forest, One-Class SVM, and regularized Mahalanobis Distance calibrated to the 99.9th percentile of healthy validation telemetry.
* **Calibrated Danger Predictor**: Advance warning classifier ($N = 10$ minutes lookahead before static SCADA threshold breach) tuned to a strict false-alarm budget ($\le 2$ alarms/day).

---

## 2. Benchmark Results: Machine Learning vs. Traditional Baselines

All models evaluated on the identical held-out test split (~30,163 observations across nodes `GT-01`, `GT-02`, and `GT-03`):

| Model Architecture | Paradigm | Test Accuracy | Macro-F1 | Model Size | Inference Latency | Edge Deployability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HistGradientBoosting (Best)**| Gradient Boosted Decision Trees | **92.44%** | **0.9206** | **1,144 KB** | **15.3 ms** | **Native ESP32 C array (<1 ms)** |
| **Random Forest** | Bagged Trees (100 estimators)| **92.86%** | **0.9253** | ~4.2 MB | 1.82 ms | Highly feasible (treelite) |
| **Static SCADA Limits** | Hard High/Low Alarm Trips | 57.10% | 0.4412 | 0 KB | <0.001 ms | Late warning (post-rupture) |
| **Rate-of-Change (ROC) Rule** | Dynamic \|dP/dt\| Threshold | 54.88% | 0.4285 | 0 KB | <0.001 ms | Susceptible to sensor noise |

---

## 3. Event Detection Performance by Fault Size

| Event Type & Size Band | Events Caught / Total | Detection Rate (%) | Median Delay After Onset | Advance Lead Time Before SCADA Trip |
| :--- | :--- | :--- | :--- | :--- |
| **Leak $\le 2\%$ Flow** | 4/4 | **100.0%** | 95.0 s | **512.5 s (~8.5 min lead)** |
| **Leak $2 - 10\%$ Flow** | 9/9 | **100.0%** | 85.0 s | **540.0 s (~9.0 min lead)** |
| **Leak $> 10\%$ Flow** | 5/5 | **100.0%** | 60.0 s | **480.0 s (~8.0 min lead)** |
| **Blockage $\le 30\%$ Area**| 3/3 | **100.0%** | 110.0 s | **620.0 s (~10.3 min lead)** |
| **Blockage $> 30\%$ Area**| 10/10 | **100.0%** | 75.0 s | **585.0 s (~9.8 min lead)** |

---

## 4. Feature Engineering (Strictly Non-Leaking 60s Windows)

Window length = $60\text{ s} / 5\text{ s} = 12$ consecutive sensor steps.
Input features are **strictly restricted to the 6 physical telemetry channels**:
`['pressure_bar_g', 'flow_mmscmd', 'gas_temp_c', 'vibration_mm_s_rms', 'ultrasonic_db_rel', 'gas_lel_pct']`

*All target labels, alarms, fault sizes, event IDs, and timestamps are stripped prior to feature engineering.*

1. **Raw Window Statistics (18 Features)**:
   - Rolling Mean, Standard Deviation, and Least-Squares Slope for each sensor.
2. **Flow-Compensated Residuals (9 Features)**:
   - Regressions $P(Q)$, $Vibr(Q)$, and $Ultra(Q)$ fitted *strictly on healthy training telemetry*.
   - Evaluates residual deviations $r_P = P - \hat{P}(Q)$, $r_V = V - \hat{V}(Q)$, $r_U = U - \hat{U}(Q)$ (Mean, Std, Slope).
3. **Smoothed Dynamics / Lag (12 Features)**:
   - Half-window differences ($[t_6 \dots t_{11}] - [t_0 \dots t_5]$) and total step deltas ($X_t - X_{t-11}$).
4. **Total Feature Count**: **39 non-leaking, physics-grounded features**.

---

## 5. Quick Start & Execution

```bash
# 1. Install dependencies (TensorFlow completely removed)
pip install -r requirements.txt

# 2. Run full training, ablation, multi-seed & stress test pipeline
cd lpg_pipeline_monitor
python run_pipeline.py

# 3. Launch interactive SCADA telemetry dashboard
streamlit run dashboard.py

# 4. Or open Jupyter Notebook
jupyter notebook ../pipeline_ml.ipynb
```

---

## 6. Live Edge Inference API: `predict_window(df_window)`

For real-time RTU / ESP32 sensor integration:

```python
from predict_service import predict_window
import pandas as pd

# Pass a 12-row window (60 seconds) of sensor readings
window_df = pd.read_csv('live_telemetry_sample.csv') 
result = predict_window(window_df)

print(result)
# Output:
# {
#   'level': 'Warning',
#   'likely_cause': 'Leak',
#   'probability': 0.9995,
#   'danger_probability': 0.0154,
#   'fault_probabilities': {'Normal': 0.0001, 'Leak': 0.9995, 'Blockage': 0.0004}
# }
```

---

## 7. Directory Structure

```
FYP/
├── pipeline_training_data.csv          Raw pipeline telemetry (207,360 rows)
├── pipeline_ml.ipynb                   Interactive end-to-end Jupyter Notebook
├── CHANGES.md                          Audit log of all deleted and modified files
├── results/
│   ├── summary.csv                     Machine-readable evaluation benchmark
│   ├── summary.md                      Comprehensive scientific report
│   ├── ablation_study.csv              Feature ablation results (18F vs 27F vs 30F vs 39F)
│   ├── multi_seed_metrics.csv          5-seed evaluation table (Mean +- Std)
│   ├── shap_summary.png                Tree SHAP feature importance plot
│   └── reliability_diagram.png         Probability calibration diagram
└── lpg_pipeline_monitor/
    ├── ml_pipeline.py                  Vectorized feature extractor & models
    ├── train_and_evaluate.py           Training & evaluation harness
    ├── run_full_evaluation.py          Benchmark runner
    ├── run_pipeline.py                 One-click pipeline runner
    ├── predict_service.py              Edge predict_window service
    ├── dashboard.py                    Streamlit live monitoring dashboard
    ├── config.py                       Pipeline configurations
    ├── requirements.txt                Cleaned dependencies (no TF/Keras)
    └── models/
        ├── best_model.joblib           Trained HistGradientBoosting model
        ├── danger_model.joblib         Calibrated Danger Predictor
        ├── random_forest.joblib        Random Forest model
        ├── residual_model.joblib       Flow residual regression weights
        └── unsupervised_and_thresholds.joblib
```
