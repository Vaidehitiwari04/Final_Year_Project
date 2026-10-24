import pandas as pd
import numpy as np
import json

df = pd.read_csv('pipeline_training_data.csv')

report = {}
report['shape'] = list(df.shape)
report['columns'] = list(df.columns)
report['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
report['missing_values'] = {col: int(val) for col, val in df.isnull().sum().items()}
report['duplicates'] = int(df.duplicated().sum())

# Categorical counts
report['label_counts'] = {str(k): int(v) for k, v in df['label'].value_counts().items()}
report['fault_type_counts'] = {str(k): int(v) for k, v in df['fault_type'].value_counts().items()}
report['split_counts'] = {str(k): int(v) for k, v in df['split'].value_counts().items()}
report['node_id_counts'] = {str(k): int(v) for k, v in df['node_id'].value_counts().items()}
if 'event_kind' in df.columns:
    report['event_kind_counts'] = {str(k): int(v) for k, v in df['event_kind'].fillna('normal').value_counts().items()}

# Crosstabs
crosstab_label_fault = pd.crosstab(df['label'], df['fault_type']).to_dict()
report['crosstab_label_fault'] = crosstab_label_fault

# Numerical features summary
sensor_cols = ['pressure_bar_g', 'flow_mmscmd', 'gas_temp_c', 'vibration_mm_s_rms', 'ultrasonic_db_rel', 'gas_lel_pct']
sensor_summary = {}
for col in sensor_cols:
    s = df[col].dropna()
    sensor_summary[col] = {
        'mean': float(s.mean()),
        'std': float(s.std()),
        'min': float(s.min()),
        '25%': float(s.quantile(0.25)),
        '50%': float(s.quantile(0.50)),
        '75%': float(s.quantile(0.75)),
        'max': float(s.max())
    }
report['sensor_summary'] = sensor_summary

# Groupby stats by fault_type
report['by_fault_type'] = {}
for ft, grp in df.groupby('fault_type'):
    report['by_fault_type'][ft] = {col: float(grp[col].dropna().mean()) for col in sensor_cols}

# Groupby stats by label
report['by_label'] = {}
for lbl, grp in df.groupby('label'):
    report['by_label'][str(lbl)] = {col: float(grp[col].dropna().mean()) for col in sensor_cols}

# Timestamps & sequentiality
df['timestamp'] = pd.to_datetime(df['timestamp'])
report['timestamp_min'] = str(df['timestamp'].min())
report['timestamp_max'] = str(df['timestamp'].max())
report['total_duration_hours'] = float((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600.0)

# Check sampling step per node
sample_steps = {}
for node in df['node_id'].unique():
    node_df = df[df['node_id'] == node].sort_values('timestamp')
    diffs = node_df['timestamp'].diff().dropna().dt.total_seconds()
    sample_steps[node] = {
        'mode_seconds': float(diffs.mode()[0]) if len(diffs.mode()) > 0 else None,
        'min_seconds': float(diffs.min()),
        'max_seconds': float(diffs.max())
    }
report['sample_steps'] = sample_steps

with open('dataset_analysis_raw.json', 'w') as f:
    json.dump(report, f, indent=2)

print("Analysis script finished successfully!")
