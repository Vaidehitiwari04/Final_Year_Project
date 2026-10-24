"""
config.py — Central configuration for the LPG Pipeline Monitoring System.
All scripts import from here to ensure consistency.
"""

# ── Sensor / Feature columns ──────────────────────────────────────────────────
SENSOR_COLS = [
    'pressure_bar_g',
    'flow_mmscmd',
    'gas_temp_c',
    'vibration_mm_s_rms',
    'ultrasonic_db_rel',
    'gas_lel_pct',
]
SENSOR_UNITS = {
    'pressure_bar_g':      'bar(g)',
    'flow_mmscmd':         'MMSCMD',
    'gas_temp_c':          '°C',
    'vibration_mm_s_rms':  'mm/s RMS',
    'ultrasonic_db_rel':   'dB(rel)',
    'gas_lel_pct':         '% LEL',
}
SENSOR_LABELS = {
    'pressure_bar_g':      'Pressure',
    'flow_mmscmd':         'Flow Rate',
    'gas_temp_c':          'Gas Temp',
    'vibration_mm_s_rms':  'Vibration',
    'ultrasonic_db_rel':   'Ultrasonic',
    'gas_lel_pct':         'Gas LEL',
}
N_FEATURES = len(SENSOR_COLS)

# ── Sequence / Window settings ────────────────────────────────────────────────
SEQ_LEN     = 10    # 10 × 5 s = 50-second sliding window
STEP_SIZE   = 1     # stride for sliding window

# ── Target encodings ──────────────────────────────────────────────────────────
FAULT_TYPE_MAP    = {'none': 0, 'leak': 1, 'blockage': 2}
FAULT_TYPE_INV    = {0: 'none',    1: 'leak',    2: 'blockage'}
FAULT_TYPE_LABELS = ['Normal',     'Leak',        'Blockage']
N_FAULT_CLASSES   = 3

SEVERITY_MAP    = {0: 'Normal',  1: 'Warning',  2: 'Danger'}
SEVERITY_INV    = {0: 'Normal',  1: 'Warning',  2: 'Danger'}
SEVERITY_COLORS = {0: '#22c55e', 1: '#f59e0b', 2: '#ef4444'}
N_LABEL_CLASSES = 3

# ── Training hyper-parameters ─────────────────────────────────────────────────
BATCH_SIZE    = 256
EPOCHS        = 60
LEARNING_RATE = 1e-3
PATIENCE      = 8   # EarlyStopping patience (epochs)
LR_PATIENCE   = 4   # ReduceLROnPlateau patience

# ── File paths (relative to lpg_pipeline_monitor/ directory) ─────────────────
DATA_CSV     = 'data/pipeline_training_data.csv'
SPLITS_DIR   = 'data/splits'
MODELS_DIR   = 'models'
RESULTS_DIR  = 'results'
SCALER_PATH  = 'models/scaler.pkl'
MODEL_PATH   = 'models/best_model.keras'
HISTORY_PATH = 'results/training_history.json'
REPORT_PATH  = 'results/evaluation_report.json'
