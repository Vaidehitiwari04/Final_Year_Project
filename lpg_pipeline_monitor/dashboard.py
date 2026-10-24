"""
dashboard.py — Streamlit real-time monitoring dashboard for the LPG Pipeline
               Monitoring & Fault Detection System.

Run:
    cd lpg_pipeline_monitor
    streamlit run dashboard.py

Tabs:
    🔴 Live Monitor   — Replay test-set telemetry with live CNN-LSTM inference
    📊 Model Results  — Training curves + confusion matrices + classification report
    📈 Data Analysis  — Dataset statistics, sensor distributions, fault signatures
    ℹ️  About         — Architecture overview, project description
"""

import os
import json
import pickle
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title='LPG Pipeline Monitor',
    page_icon='🔥',
    layout='wide',
    initial_sidebar_state='expanded',
)

from config import (
    SENSOR_COLS, SENSOR_LABELS, SENSOR_UNITS,
    FAULT_TYPE_LABELS, FAULT_TYPE_INV, FAULT_TYPE_MAP,
    SEVERITY_MAP, SEVERITY_COLORS,
    MODEL_PATH, SCALER_PATH, RESULTS_DIR, SPLITS_DIR,
    DATA_CSV, HISTORY_PATH, REPORT_PATH, SEQ_LEN,
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main { background-color: #0a0e1a; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0f1729 50%, #0a0e1a 100%); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #111827 100%);
    border-right: 1px solid #1e293b;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin: 4px 0;
}
.metric-label {
    font-size: 0.78rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 500;
}
.metric-sublabel {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 2px;
}

/* Status badges */
.badge-normal  { background: linear-gradient(90deg,#14532d,#166534); color:#4ade80; border:1px solid #166534; }
.badge-warning { background: linear-gradient(90deg,#713f12,#92400e); color:#fbbf24; border:1px solid #92400e; }
.badge-danger  { background: linear-gradient(90deg,#7f1d1d,#991b1b); color:#f87171; border:1px solid #991b1b; }
.badge-none    { background: linear-gradient(90deg,#1e3a5f,#1e40af); color:#60a5fa; border:1px solid #1e40af; }
.badge-leak    { background: linear-gradient(90deg,#7f1d1d,#991b1b); color:#f87171; border:1px solid #991b1b; }
.badge-blockage{ background: linear-gradient(90deg,#4c1d95,#6d28d9); color:#c4b5fd; border:1px solid #6d28d9; }

.badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* Node card */
.node-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px;
}

/* Alert box */
.alert-critical {
    background: linear-gradient(90deg, rgba(127,29,29,0.4), rgba(153,27,27,0.2));
    border: 1px solid #991b1b;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 12px 16px;
    color: #fca5a5;
    margin: 8px 0;
    animation: pulse-red 2s infinite;
}
.alert-warning {
    background: linear-gradient(90deg, rgba(120,53,15,0.4), rgba(146,64,14,0.2));
    border: 1px solid #92400e;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 12px 16px;
    color: #fcd34d;
    margin: 8px 0;
}
@keyframes pulse-red {
    0%, 100% { border-left-color: #ef4444; }
    50%       { border-left-color: #fca5a5; }
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #94a3b8;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom-color: #38bdf8 !important;
}

/* Title */
.main-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}
.main-subtitle {
    color: #64748b;
    font-size: 0.9rem;
    margin-bottom: 24px;
}

.live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ef4444;
    animation: blink 1s step-start infinite;
    margin-right: 6px;
}
@keyframes blink { 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner='Loading CNN-LSTM model …')
def load_model():
    import tensorflow as tf
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource(show_spinner='Loading scaler …')
def load_scaler():
    if not os.path.exists(SCALER_PATH):
        return None
    with open(SCALER_PATH, 'rb') as f:
        return pickle.load(f)


@st.cache_data(show_spinner='Loading dataset …')
def load_test_csv():
    if not os.path.exists(DATA_CSV):
        return None
    df = pd.read_csv(DATA_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df[df['split'] == 'test'].reset_index(drop=True)


@st.cache_data(show_spinner='Loading training history …')
def load_history():
    if not os.path.exists(HISTORY_PATH):
        return None
    with open(HISTORY_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner='Loading evaluation report …')
def load_report():
    if not os.path.exists(REPORT_PATH):
        return None
    with open(REPORT_PATH) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def predict_window(model, scaler, window_df: pd.DataFrame):
    """
    Run inference on a SEQ_LEN-row window.
    Returns (fault_idx, fault_proba, label_idx, label_proba) or None.
    """
    if model is None or scaler is None:
        return None
    if len(window_df) < SEQ_LEN:
        return None
    vals = window_df[SENSOR_COLS].values[-SEQ_LEN:]
    if np.isnan(vals).any():
        return None
    scaled = scaler.transform(vals).astype(np.float32)
    inp = scaled[np.newaxis, ...]  # (1, seq_len, n_features)
    pf, pl = model.predict(inp, verbose=0)
    return int(np.argmax(pf[0])), pf[0], int(np.argmax(pl[0])), pl[0]


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

DARK_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(15,23,42,0.6)',
    font=dict(color='#94a3b8', family='Inter'),
    xaxis=dict(gridcolor='#1e293b', showgrid=True, zeroline=False),
    yaxis=dict(gridcolor='#1e293b', showgrid=True, zeroline=False),
    margin=dict(l=40, r=20, t=40, b=40),
)
SENSOR_PALETTE = ['#38bdf8','#818cf8','#4ade80','#f59e0b','#f87171','#a78bfa']


def sensor_line_chart(history_df: pd.DataFrame, node_id: str, n_points: int = 120):
    """Multi-trace line chart for all 6 sensors for a given node."""
    node_df = history_df[history_df['node_id'] == node_id].tail(n_points)
    fig = make_subplots(rows=3, cols=2, shared_xaxes=True,
                        subplot_titles=[SENSOR_LABELS[c] for c in SENSOR_COLS],
                        vertical_spacing=0.08, horizontal_spacing=0.06)
    positions = [(1,1),(1,2),(2,1),(2,2),(3,1),(3,2)]
    for i, col in enumerate(SENSOR_COLS):
        row, c = positions[i]
        fig.add_trace(
            go.Scatter(
                x=node_df['timestamp'], y=node_df[col],
                mode='lines', name=SENSOR_LABELS[col],
                line=dict(color=SENSOR_PALETTE[i], width=1.5),
                fill='tozeroy',
                fillcolor=SENSOR_PALETTE[i].replace(')', ',0.08)').replace('rgb', 'rgba') if 'rgb' in SENSOR_PALETTE[i] else SENSOR_PALETTE[i] + '14',
            ),
            row=row, col=c,
        )
        fig.update_yaxes(title_text=SENSOR_UNITS[col], row=row, col=c,
                         gridcolor='#1e293b', tickfont=dict(size=9))
    fig.update_layout(
        height=480, showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.5)',
        font=dict(color='#94a3b8', size=10, family='Inter'),
        margin=dict(l=50, r=10, t=40, b=30),
        title=dict(text=f'Sensor Telemetry — {node_id}', font=dict(size=14, color='#e2e8f0')),
    )
    for ann in fig.layout.annotations:
        ann.font.color = '#94a3b8'
        ann.font.size = 10
    return fig


def confusion_heatmap(cm: list, labels: list, title: str):
    cm_arr = np.array(cm)
    cm_norm = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True).clip(min=1)
    fig = go.Figure(go.Heatmap(
        z=cm_norm, x=[f'Pred {l}' for l in labels], y=[f'True {l}' for l in labels],
        colorscale='Blues', text=cm_arr,
        texttemplate='%{text}',
        hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{text}<br>Rate: %{z:.2%}<extra></extra>',
        showscale=True,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color='#e2e8f0', size=13)),
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.5)',
        font=dict(color='#94a3b8', family='Inter'),
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig


def training_curves_fig(history: dict):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Fault-Type Accuracy', 'Total Loss'])
    # Accuracy
    fig.add_trace(go.Scatter(
        y=history['fault_type_accuracy'], mode='lines',
        name='Train Acc', line=dict(color='#38bdf8', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(
        y=history['val_fault_type_accuracy'], mode='lines',
        name='Val Acc', line=dict(color='#818cf8', width=2, dash='dash')), row=1, col=1)
    # Loss
    fig.add_trace(go.Scatter(
        y=history['loss'], mode='lines',
        name='Train Loss', line=dict(color='#4ade80', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(
        y=history['val_loss'], mode='lines',
        name='Val Loss', line=dict(color='#f59e0b', width=2, dash='dash')), row=1, col=2)
    fig.update_layout(
        height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.5)',
        font=dict(color='#94a3b8', family='Inter', size=11),
        margin=dict(l=40, r=20, t=50, b=30),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
    )
    for ann in fig.layout.annotations:
        ann.font.color = '#e2e8f0'; ann.font.size = 12
    fig.update_xaxes(gridcolor='#1e293b', title_text='Epoch')
    fig.update_yaxes(gridcolor='#1e293b')
    return fig


def probability_bar(proba: np.ndarray, labels: list, colors: list):
    fig = go.Figure(go.Bar(
        x=labels, y=proba,
        marker_color=colors,
        text=[f'{p:.1%}' for p in proba],
        textposition='outside',
        textfont=dict(size=12, color='#e2e8f0'),
    ))
    fig.update_layout(
        yaxis=dict(range=[0, 1.15], gridcolor='#1e293b', tickformat='.0%'),
        xaxis=dict(gridcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.5)',
        font=dict(color='#94a3b8', family='Inter'),
        margin=dict(l=20, r=20, t=20, b=20),
        height=180,
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 10px;'>
        <div style='font-size:2.5rem'>🔥</div>
        <div style='font-size:1.1rem; font-weight:700; color:#e2e8f0;'>LPG Pipeline</div>
        <div style='font-size:0.8rem; color:#64748b;'>Monitoring System</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("**🖥️ Node Selection**")
    selected_node = st.selectbox(
        'Monitoring Station', ['GT-01', 'GT-02', 'GT-03'], label_visibility='collapsed')

    st.markdown("**⚡ Simulation Speed**")
    step_size = st.slider('Steps per click', 1, 20, 5, label_visibility='collapsed')

    st.divider()

    # Model status
    model = load_model()
    scaler = load_scaler()
    model_ok = model is not None
    scaler_ok = scaler is not None

    st.markdown("**🤖 System Status**")
    model_status = "✅ Loaded" if model_ok else "⚠️ Not found"
    scaler_status = "✅ Loaded" if scaler_ok else "⚠️ Not found"
    st.markdown(f"""
    <div style='font-size:0.8rem; color:#94a3b8; line-height:2;'>
        Model: <span style='color:{"#4ade80" if model_ok else "#f59e0b"}'>{model_status}</span><br>
        Scaler: <span style='color:{"#4ade80" if scaler_ok else "#f59e0b"}'>{scaler_status}</span><br>
        Window: {SEQ_LEN} steps × 5s = 50s<br>
        Dataset: 207,360 rows
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='font-size:0.72rem; color:#475569; text-align:center;'>
        FYP — CNN-LSTM Fault Detection<br>
        LPG Transmission Pipeline
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class='main-title'>🔥 LPG Pipeline Monitoring System</div>
<div class='main-subtitle'>CNN-LSTM Real-Time Fault Detection & Severity Classification</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_live, tab_results, tab_data, tab_about = st.tabs([
    '🔴 Live Monitor',
    '📊 Model Results',
    '📈 Data Analysis',
    'ℹ️ About',
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════

with tab_live:

    test_df = load_test_csv()

    if test_df is None:
        st.warning(f"Test dataset not found at '{DATA_CSV}'. Run preprocessing first.")
        st.stop()

    node_df = test_df[test_df['node_id'] == selected_node].sort_values('timestamp').reset_index(drop=True)
    total_rows = len(node_df)

    # Session state for playback
    if 'frame_idx' not in st.session_state:
        st.session_state.frame_idx = SEQ_LEN

    frame = min(st.session_state.frame_idx, total_rows)

    # ── Control bar ───────────────────────────────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 1, 3])
    with ctrl_col1:
        if st.button('▶ Next Step', use_container_width=True, type='primary'):
            st.session_state.frame_idx = min(frame + step_size, total_rows)
            st.rerun()
    with ctrl_col2:
        if st.button('⏭ Jump +100', use_container_width=True):
            st.session_state.frame_idx = min(frame + 100, total_rows)
            st.rerun()
    with ctrl_col3:
        if st.button('⏮ Reset', use_container_width=True):
            st.session_state.frame_idx = SEQ_LEN
            st.rerun()
    with ctrl_col4:
        progress_pct = frame / total_rows
        st.progress(progress_pct, text=f'Replaying test set: row {frame}/{total_rows}  ({progress_pct:.1%})')

    st.divider()

    # ── Current window ────────────────────────────────────────────────────────
    window = node_df.iloc[max(0, frame - SEQ_LEN):frame]
    current_row = node_df.iloc[frame - 1]
    true_fault = FAULT_TYPE_INV[int(current_row['fault_type'].strip() if isinstance(current_row['fault_type'], str) else 0)]
    true_label = SEVERITY_MAP[int(current_row['label'])]

    # Run inference
    result = predict_window(model, scaler, window)

    # Top metrics row
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    sensor_vals = {col: current_row[col] for col in SENSOR_COLS}
    sensor_colors = ['#38bdf8','#818cf8','#4ade80','#f59e0b','#f87171','#a78bfa']

    for i, (col, mc) in enumerate(zip(SENSOR_COLS, [mc1,mc2,mc3,mc4,mc5,mc6])):
        with mc:
            val = sensor_vals[col]
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{SENSOR_LABELS[col]}</div>
                <div class='metric-value' style='color:{sensor_colors[i]};font-size:1.5rem'>
                    {val:.2f}
                </div>
                <div class='metric-sublabel'>{SENSOR_UNITS[col]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    # ── Prediction + truth panel ──────────────────────────────────────────────
    pred_col, truth_col, prob_col = st.columns([1, 1, 2])

    with pred_col:
        st.markdown("<div class='section-header'>🤖 Model Prediction</div>", unsafe_allow_html=True)
        if result:
            f_idx, f_prob, l_idx, l_prob = result
            pred_fault_name = FAULT_TYPE_LABELS[f_idx].lower()
            pred_label_name = SEVERITY_MAP[l_idx].lower()
            sev_color = SEVERITY_COLORS[l_idx]

            st.markdown(f"""
            <div class='node-card' style='border-color:{sev_color}33;'>
                <div style='margin-bottom:12px'>
                    <div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>Fault Type</div>
                    <span class='badge badge-{pred_fault_name}'>{FAULT_TYPE_LABELS[f_idx]}</span>
                    <span style='font-size:0.8rem;color:#64748b;margin-left:8px'>{f_prob[f_idx]:.1%}</span>
                </div>
                <div>
                    <div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>Severity</div>
                    <span class='badge badge-{pred_label_name}'>{SEVERITY_MAP[l_idx]}</span>
                    <span style='font-size:0.8rem;color:#64748b;margin-left:8px'>{l_prob[l_idx]:.1%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Alert if danger
            if l_idx == 2:
                st.markdown(f"""
                <div class='alert-critical'>
                    <span class='live-dot'></span>
                    <strong>CRITICAL FAULT DETECTED</strong><br>
                    {FAULT_TYPE_LABELS[f_idx]} on {selected_node} — Immediate action required!
                </div>
                """, unsafe_allow_html=True)
            elif l_idx == 1:
                st.markdown(f"""
                <div class='alert-warning'>
                    ⚠️ <strong>WARNING</strong> — Early-stage {FAULT_TYPE_LABELS[f_idx].lower()} detected on {selected_node}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info('Run the training pipeline first to enable live inference.')

    with truth_col:
        st.markdown("<div class='section-header'>📋 Ground Truth</div>", unsafe_allow_html=True)
        true_fault_raw = str(current_row['fault_type']).strip()
        true_label_raw = int(current_row['label'])
        st.markdown(f"""
        <div class='node-card'>
            <div style='margin-bottom:12px'>
                <div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>True Fault</div>
                <span class='badge badge-{true_fault_raw}'>{true_fault_raw.capitalize()}</span>
            </div>
            <div style='margin-bottom:12px'>
                <div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>True Severity</div>
                <span class='badge badge-{SEVERITY_MAP[true_label_raw].lower()}'>{SEVERITY_MAP[true_label_raw]}</span>
            </div>
            <div style='margin-top:8px;font-size:0.75rem;color:#64748b'>
                🕐 {current_row['timestamp']}<br>
                📍 Node: {selected_node}<br>
                ⚡ Fault size: {current_row['fault_size_pct']:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with prob_col:
        st.markdown("<div class='section-header'>📊 Prediction Probabilities</div>", unsafe_allow_html=True)
        if result:
            f_idx, f_prob, l_idx, l_prob = result
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.markdown("**Fault Type**")
                st.plotly_chart(probability_bar(
                    f_prob, FAULT_TYPE_LABELS,
                    colors=['#60a5fa','#f87171','#c4b5fd']), use_container_width=True, key='prob_fault')
            with pcol2:
                st.markdown("**Severity**")
                st.plotly_chart(probability_bar(
                    l_prob, [SEVERITY_MAP[i] for i in range(3)],
                    colors=['#4ade80','#fbbf24','#f87171']), use_container_width=True, key='prob_label')
        else:
            st.info('Awaiting model predictions …')

    st.divider()

    # ── Sensor charts ─────────────────────────────────────────────────────────
    history_window = node_df.iloc[max(0, frame - 120):frame]
    st.plotly_chart(sensor_line_chart(history_window, selected_node), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════

with tab_results:
    history = load_history()
    report  = load_report()

    if history is None and report is None:
        st.info('Training and evaluation not yet run. Execute `python train.py` then `python evaluate.py`.')
    else:
        if history:
            st.markdown("### 📉 Training Curves")
            st.plotly_chart(training_curves_fig(history), use_container_width=True)
            n_epochs = len(history['loss'])
            best_epoch = int(np.argmax(history['val_fault_type_accuracy'])) + 1
            best_val_acc = max(history['val_fault_type_accuracy'])
            mc1, mc2, mc3, mc4 = st.columns(4)
            for col, lbl, val in zip(
                [mc1, mc2, mc3, mc4],
                ['Epochs Trained', 'Best Epoch', 'Best Val Acc', 'Min Val Loss'],
                [n_epochs, best_epoch, f'{best_val_acc:.4f}',
                 f'{min(history["val_loss"]):.4f}']
            ):
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>{lbl}</div>
                        <div class='metric-value' style='color:#38bdf8;font-size:1.6rem'>{val}</div>
                    </div>""", unsafe_allow_html=True)

        st.divider()

        if report:
            st.markdown("### 🎯 Test Set Performance")
            ft = report['fault_type']
            sl = report['severity_label']

            # Summary metrics
            rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)
            metrics_data = [
                ('Fault Accuracy', f"{ft['accuracy']:.4f}", '#38bdf8'),
                ('Fault F1 (W)',   f"{ft['f1_weighted']:.4f}", '#818cf8'),
                ('Fault AUC',      f"{ft['auc_roc']:.4f}" if ft['auc_roc'] else 'N/A', '#4ade80'),
                ('Sev. Accuracy',  f"{sl['accuracy']:.4f}", '#f59e0b'),
                ('Sev. F1 (W)',    f"{sl['f1_weighted']:.4f}", '#f87171'),
                ('Sev. AUC',       f"{sl['auc_roc']:.4f}" if sl['auc_roc'] else 'N/A', '#a78bfa'),
            ]
            for col, (lbl, val, clr) in zip([rc1,rc2,rc3,rc4,rc5,rc6], metrics_data):
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>{lbl}</div>
                        <div class='metric-value' style='color:{clr};font-size:1.5rem'>{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

            # Confusion matrices
            cm1, cm2 = st.columns(2)
            with cm1:
                st.plotly_chart(confusion_heatmap(
                    ft['confusion_matrix'], FAULT_TYPE_LABELS,
                    'Confusion Matrix — Fault Type'), use_container_width=True)
            with cm2:
                st.plotly_chart(confusion_heatmap(
                    sl['confusion_matrix'], [SEVERITY_MAP[i] for i in range(3)],
                    'Confusion Matrix — Severity Label'), use_container_width=True)

            # Per-class breakdown
            st.markdown("### 📋 Per-Class Breakdown")
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown("**Fault Type Classification**")
                rows = []
                for cls in FAULT_TYPE_LABELS:
                    r = ft['per_class'].get(cls, {})
                    rows.append({
                        'Class':     cls,
                        'Precision': f"{r.get('precision', 0):.4f}",
                        'Recall':    f"{r.get('recall', 0):.4f}",
                        'F1-Score':  f"{r.get('f1-score', 0):.4f}",
                        'Support':   int(r.get('support', 0)),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            with tc2:
                st.markdown("**Severity Label Classification**")
                rows = []
                for cls in [SEVERITY_MAP[i] for i in range(3)]:
                    r = sl['per_class'].get(cls, {})
                    rows.append({
                        'Class':     cls,
                        'Precision': f"{r.get('precision', 0):.4f}",
                        'Recall':    f"{r.get('recall', 0):.4f}",
                        'F1-Score':  f"{r.get('f1-score', 0):.4f}",
                        'Support':   int(r.get('support', 0)),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with tab_data:
    st.markdown("### 📊 Dataset Overview")

    da1, da2, da3, da4 = st.columns(4)
    overview = [
        ('Total Records',  '207,360', '#38bdf8'),
        ('Monitoring Nodes', '3 (GT-01/02/03)', '#818cf8'),
        ('Sampling Interval', '5 seconds', '#4ade80'),
        ('Coverage', '96 hours', '#f59e0b'),
    ]
    for col, (lbl, val, clr) in zip([da1,da2,da3,da4], overview):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{lbl}</div>
                <div class='metric-value' style='color:{clr};font-size:1.3rem'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🎯 Class Distribution")

    dc1, dc2 = st.columns(2)
    with dc1:
        fault_data = {'none': 105071, 'leak': 58915, 'blockage': 43374}
        fig_fault = go.Figure(go.Pie(
            labels=['Normal', 'Leak', 'Blockage'],
            values=list(fault_data.values()),
            hole=0.55,
            marker_colors=['#60a5fa', '#f87171', '#c4b5fd'],
            textinfo='label+percent',
            textfont=dict(size=12, color='white'),
        ))
        fig_fault.update_layout(
            title=dict(text='Fault Type Distribution', font=dict(color='#e2e8f0', size=13)),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='Inter'),
            showlegend=False, height=300, margin=dict(l=0,r=0,t=50,b=0),
        )
        st.plotly_chart(fig_fault, use_container_width=True)

    with dc2:
        sev_data = {'Normal': 105071, 'Warning': 19618, 'Danger': 82671}
        fig_sev = go.Figure(go.Pie(
            labels=list(sev_data.keys()),
            values=list(sev_data.values()),
            hole=0.55,
            marker_colors=['#4ade80', '#fbbf24', '#f87171'],
            textinfo='label+percent',
            textfont=dict(size=12, color='white'),
        ))
        fig_sev.update_layout(
            title=dict(text='Severity Label Distribution', font=dict(color='#e2e8f0', size=13)),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='Inter'),
            showlegend=False, height=300, margin=dict(l=0,r=0,t=50,b=0),
        )
        st.plotly_chart(fig_sev, use_container_width=True)

    st.divider()
    st.markdown("### 🔬 Fault Signature Analysis")

    sig_data = {
        'Sensor':         ['Pressure (bar)', 'Flow (MMSCMD)', 'Temp (°C)', 'Vibration (mm/s)', 'Ultrasonic (dB)', 'Gas LEL (%)'],
        'Normal':         [55.77, 10.30, 18.05, 1.82,  0.12, 0.36],
        'Leak':           [53.51,  9.58, 16.89, 2.98, 12.20, 4.39],
        'Blockage':       [60.46,  7.02, 19.07, 3.27,  5.50, 0.36],
    }
    sig_df = pd.DataFrame(sig_data).set_index('Sensor')

    fig_sig = go.Figure()
    colors = ['#60a5fa', '#f87171', '#c4b5fd']
    for col, clr in zip(['Normal', 'Leak', 'Blockage'], colors):
        # Normalize per sensor for radar-like bar chart
        fig_sig.add_trace(go.Bar(name=col, x=sig_df.index,
                                  y=sig_df[col], marker_color=clr))
    fig_sig.update_layout(
        barmode='group',
        title=dict(text='Mean Sensor Readings by Fault Type', font=dict(color='#e2e8f0', size=13)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.5)',
        font=dict(color='#94a3b8', family='Inter'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        height=320, margin=dict(l=40,r=20,t=50,b=60),
        xaxis=dict(gridcolor='rgba(0,0,0,0)', tickangle=-20),
        yaxis=dict(gridcolor='#1e293b', title='Mean Value'),
    )
    st.plotly_chart(fig_sig, use_container_width=True)

    st.markdown("### 📁 Dataset Splits")
    splits_info = pd.DataFrame({
        'Split': ['train', 'val', 'test'],
        'Records': [146377, 30820, 30163],
        'Percentage': ['70.59%', '14.86%', '14.55%'],
        'Events': [245, 46, 44],
        'Event Overlap': ['None', 'None', 'None'],
    })
    st.dataframe(splits_info, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════

with tab_about:
    ab1, ab2 = st.columns([3, 2])

    with ab1:
        st.markdown("""
        ### 🏗️ Project Architecture

        **LPG Pipeline Monitoring & Fault Prediction System**  
        Final Year Project — Deep Learning for Industrial IoT Safety

        ---

        #### 🎯 Objective
        Design and implement a real-time fault detection system for LPG transmission pipelines
        using a CNN-LSTM deep learning model that classifies sensor telemetry into:
        - **Fault Type**: Normal / Leak / Blockage
        - **Severity**: Normal / Warning / Danger

        #### 🧠 Model Architecture
        ```
        Input: (batch, T=10 steps, 6 sensors)
            ↓
        Causal Conv1D × 3  [64 → 128 → 64 filters]
        BatchNorm + Dropout(0.3)
            ↓
        Bidirectional LSTM × 2  [64 → 32 units]
        Dropout(0.3)
            ↓
        Shared Dense [128 → 64]
            ↓
        ┌─────────────────┐    ┌─────────────────┐
        │ Fault Head      │    │ Severity Head   │
        │ Dense(32)       │    │ Dense(32)       │
        │ Softmax(3)      │    │ Softmax(3)      │
        │ none/leak/block │    │ Normal/Warn/Dang│
        └─────────────────┘    └─────────────────┘
        ```

        #### 📐 Why CNN-LSTM?
        | Component | Role |
        |---|---|
        | **Causal Conv1D** | Extracts local temporal patterns (pressure wave steepness dP/dt, acoustic burst onset) without future leakage |
        | **Bidirectional LSTM** | Models the long-term sequential trajectory — distinguishes transient demand steps from progressive faults |
        | **Dual Output** | Simultaneous fault identification (what) and severity staging (how bad) |
        """)

    with ab2:
        st.markdown("""
        ### 📡 Sensor Features

        | Feature | Physical Meaning |
        |---|---|
        | `pressure_bar_g` | In-line static gauge pressure |
        | `flow_mmscmd` | Gas volumetric flow rate |
        | `gas_temp_c` | Fluid temperature (Joule-Thomson effect) |
        | `vibration_mm_s_rms` | Mechanical pipe vibration |
        | `ultrasonic_db_rel` | High-frequency acoustic emission |
        | `gas_lel_pct` | Atmospheric gas concentration (% LEL) |

        ### 📊 Dataset Summary

        | Property | Value |
        |---|---|
        | Records | 207,360 |
        | Nodes | GT-01, GT-02, GT-03 |
        | Duration | 96 hours |
        | Sampling | 5 seconds uniform |
        | Features | 16 columns |
        | Targets | fault_type + label |
        | Duplicates | 0 |
        | Event overlap | 0 |

        ### 🔬 Fault Signatures
        - **Leak** → pressure ↓, flow ↓ slightly, Joule-Thomson cooling, massive ultrasonic spike, LEL spike
        - **Blockage** → pressure ↑ upstream, flow ↓ significantly, slight temperature rise
        """)

    st.divider()
    st.markdown("""
    ### 🔄 Pipeline Flow

    ```
    pipeline_training_data.csv
            │
            ▼
    preprocess.py  →  data/splits/*.npy  +  models/scaler.pkl
            │
            ▼
    train.py       →  models/best_model.keras  +  results/training_history.json
            │
            ▼
    evaluate.py    →  results/evaluation_report.json  +  results/confusion_*.csv
            │
            ▼
    dashboard.py   →  🌐 Streamlit Live Monitoring Dashboard
    ```
    """)
