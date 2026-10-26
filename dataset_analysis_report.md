# LPG Pipeline Monitoring Dataset Analysis Report

**Project**: LPG Pipeline Monitoring and Fault Prediction System using CNN-LSTM  
**Dataset Analyzed**: `pipeline_training_data.csv`  
**Total Records**: 207,360  
**Total Features/Columns**: 16  
**Time Coverage**: 2026-01-01 00:00:00 to 2026-01-04 23:59:55 (96.00 continuous hours)  
**Sampling Interval**: Uniform 5.0 seconds per monitoring station  

---

## 1. Dataset Overview & Schema

The dataset represents multi-sensor telemetry recorded across three pipeline monitoring stations (`GT-01`, `GT-02`, and `GT-03`) along an LPG (Liquefied Petroleum Gas) transmission line.

### Column Inventory & Physical Definitions

| Column Name | Data Type | Physical Meaning | Unit / Scale | Missing Count | Missing % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `timestamp` | `str` (datetime) | Telemetry observation time (5s interval) | YYYY-MM-DD HH:MM:SS | 0 | 0.00% |
| `node_id` | `str` | Pipeline monitoring node ID (`GT-01`, `GT-02`, `GT-03`) | Categorical identifier | 0 | 0.00% |
| `pressure_bar_g` | `float64` | In-line static gas pressure (gauge) | bar (gauge) | 275 | 0.13% |
| `flow_mmscmd` | `float64` | Gas volumetric flow rate | MMSCMD (Million Standard m³/day) | 280 | 0.14% |
| `gas_temp_c` | `float64` | Pipeline gas fluid temperature | °C | 294 | 0.14% |
| `vibration_mm_s_rms` | `float64` | Mechanical pipe vibration velocity | mm/s RMS | 0 | 0.00% |
| `ultrasonic_db_rel` | `float64` | High-frequency acoustic emission sensor | dB (relative to baseline) | 0 | 0.00% |
| `gas_lel_pct` | `float64` | Atmospheric gas leak detector | % LEL (Lower Explosive Limit) | 361 | 0.17% |
| `limit_alarm` | `int64` | Binary hard threshold trip alarm | 0 (Normal) / 1 (Alarm) | 0 | 0.00% |
| `label` | `int64` | Progression / severity stage | 0: Normal, 1: Warning, 2: Danger | 0 | 0.00% |
| `fault_type` | `str` | Primary fault classification target | `none`, `leak`, `blockage` | 0 | 0.00% |
| `fault_size_pct` | `float64` | Magnitude / orifice / obstruction ratio | % (0.0 to 100.0) | 0 | 0.00% |
| `event_id` | `str` | Unique simulated/recorded event tag | e.g. `EV-TR-0001` (Null for baseline) | 73,151 | 35.28% |
| `event_kind` | `str` | Ground-truth event category | `leak`, `blockage`, `demand_step`, etc. | 73,151 | 35.28% |
| `s_since_onset` | `float64` | Elapsed seconds since anomaly started | seconds | 73,151 | 35.28% |
| `split` | `str` | Pre-partitioned event-safe partition | `train`, `val`, `test` | 0 | 0.00% |

> [!NOTE]
> `event_id`, `event_kind`, and `s_since_onset` have missing values (73,151 rows) strictly during steady-state normal baseline operation where no anomaly event is active.

---

## 2. Target Columns & Class Labels

The dataset provides two complementary supervisory targets:

### A. Primary Fault Category: `fault_type`
* **`none`**: 105,071 records (50.67%) — Normal steady-state operation, operational flow demand steps, and benign sensor flutter.
* **`leak`**: 58,915 records (28.41%) — Pipeline pinhole or orifice rupture causing gas loss into surrounding atmosphere.
* **`blockage`**: 43,374 records (20.92%) — Hydrate formation, valve restriction, or slugging obstructing pipeline gas flow.

### B. Severity & Progression Stage: `label`
* **`0` (Normal)**: 105,071 records (50.67%) — Safe baseline. Matches `fault_type == 'none'`.
* **`1` (Warning / Early Onset)**: 19,618 records (9.46%) — Incipient fault phase ($t_{\text{onset}} \approx 252$ s average). Sensor signatures begin deviating, but limit alarms have not yet tripped.
* **`2` (Danger / Critical Fault)**: 82,671 records (39.87%) — Fully developed fault ($t_{\text{onset}} \approx 1445$ s average). Pressure drops or surges exceed critical safety thresholds, external gas concentrations rise, and hard limit alarms trigger.

### Cross-Tabulation: `label` vs `fault_type`

| `label` (Severity) | `blockage` | `leak` | `none` | Total |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Normal)** | 0 | 0 | 105,071 | **105,071** |
| **1 (Warning)** | 8,412 | 11,206 | 0 | **19,618** |
| **2 (Danger)** | 34,962 | 47,709 | 0 | **82,671** |
| **Total** | **43,374** | **58,915** | **105,071** | **207,360** |

---

## 3. Physical Sensor Distributions & Fault Signatures

| Sensor Feature | Normal Baseline (`none`) | Leak Condition (`leak`) | Blockage Condition (`blockage`) | Physical Diagnostic Signature |
| :--- | :--- | :--- | :--- | :--- |
| **Pressure (`pressure_bar_g`)** | **55.77 bar** (std: 0.81) | **53.51 bar** (std: 2.14) | **60.46 bar** (std: 4.88, max 97.5) | Pressure **drops** during leak due to fluid loss; pressure **surges** upstream during blockage. |
| **Flow Rate (`flow_mmscmd`)** | **10.30 MMSCMD** | **9.58 MMSCMD** | **7.02 MMSCMD** (min 2.31) | Flow significantly **drops** during blockage; moderate decrease during leak. |
| **Temperature (`gas_temp_c`)** | **18.05 °C** | **16.89 °C** | **19.07 °C** | Gas expansion causes **Joule-Thomson cooling** during leak; friction/compression warms gas during blockage. |
| **Vibration (`vibration_mm_s_rms`)** | **1.82 mm/s** | **2.98 mm/s** | **3.27 mm/s** | Mechanical turbulence and structural resonance increase under both fault states. |
| **Ultrasonic (`ultrasonic_db_rel`)** | **0.12 dB** | **12.20 dB** (max 31.6) | **5.50 dB** | **Sharp ultrasonic spike** indicates turbulent jet acoustic noise through leak orifice. |
| **Gas LEL (`gas_lel_pct`)** | **0.36 %** | **4.39 %** (max 41.4) | **0.36 %** | Direct confirmation of flammable gas escaping into atmospheric zone. |

---

## 4. Missing Values & Duplicate Records

* **Duplicate rows**: Exactly **0** duplicates found.
* **Missing sensor readings**:
  - `pressure_bar_g`: 275 rows (0.13%)
  - `flow_mmscmd`: 280 rows (0.14%)
  - `gas_temp_c`: 294 rows (0.14%)
  - `gas_lel_pct`: 361 rows (0.17%)
* **Mitigation strategy**: Missing sensor readings are imputed using forward-fill followed by backward-fill grouped per node, simulating continuous telemetry buffer hold. Scaling and normalization parameters (StandardScaler) are fitted exclusively on the training split to prevent information leakage.

---

## 5. Dataset Splitting & Leakage Prevention

The dataset contains a pre-partitioned `split` column structured specifically at the **event boundary level**:
* **`train`**: 146,377 records (70.59%), covering 245 distinct events.
* **`val`**: 30,820 records (14.86%), covering 46 distinct events.
* **`test`**: 30,163 records (14.55%), covering 44 distinct events.
* **Overlap across splits**: **0 event overlap**. This guarantees zero data leakage between train, validation, and test partitions.

---

## 6. Suitability for Classical Machine Learning Modeling

* **Time-Series Integrity**: Yes. The dataset consists of chronological sensor readings spaced evenly at 5.0-second intervals per monitoring node.
* **Temporal Dynamics**: Pipeline hydraulic transitions (pressure wave propagation, acoustic bursts, mass balance decay) unfold over tens of seconds.
* **Classical ML Architecture Justification**:
  1. **Physics-Informed Feature Engineering**: Explicit calculation of rolling statistics (mean, std, least-squares slope) and flow-compensated residuals ($P, V, U$ minus empirical flow regression on healthy data) extracts the exact hydraulic invariants ($P \propto f(Q)$) without requiring deep recurrent networks to learn them from noisy signals.
  2. **Gradient Boosted Decision Trees (HistGradientBoosting / Random Forest)**: Superior performance on tabular and windowed time-series features without overfitting to specific node noise baselines.
  3. **Edge Feasibility (ESP32 / Microcontrollers)**: Inference requires only ~15 ms (or microsecond-level C array execution) with tiny memory footprint (~1.1 MB vs. multi-megabyte neural runtimes that exceed ESP32 internal SRAM).
* **Sequence Window Configuration**:
  - Window Length: $W = 12$ time steps ($12 \times 5\text{ s} = 60\text{ seconds}$ sliding observation window).
  - Multi-sensor features: 6 input channels (`pressure_bar_g`, `flow_mmscmd`, `gas_temp_c`, `vibration_mm_s_rms`, `ultrasonic_db_rel`, `gas_lel_pct`). Total 39 non-leaking features.
  - Strict leakage prevention: No future lookahead in feature computation; forward/backward imputation per node.
