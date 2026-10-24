"""
run_pipeline.py — One-click runner: preprocess → train → evaluate.

Run:
    cd lpg_pipeline_monitor
    python run_pipeline.py

Then launch dashboard:
    streamlit run dashboard.py
"""

import subprocess
import sys
import time
import os


def run_step(name: str, script: str):
    print(f"\n{'=' * 60}")
    print(f"  STEP: {name}")
    print(f"{'=' * 60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], check=True)
    elapsed = time.time() - t0
    print(f"\n  ✅ {name} complete in {elapsed/60:.1f} min")
    return elapsed


def main():
    print("\n🔥 LPG Pipeline Monitoring System — Full Training Pipeline")
    print("=" * 60)
    print(f"  Working directory: {os.getcwd()}")

    total = 0
    total += run_step("Data Preprocessing", "preprocess.py")
    total += run_step("CNN-LSTM Training",  "train.py")
    total += run_step("Model Evaluation",   "evaluate.py")

    print(f"\n{'=' * 60}")
    print(f"  🎉 ALL STEPS COMPLETE in {total/60:.1f} minutes")
    print(f"  📁 Trained model  → models/best_model.keras")
    print(f"  📁 Results        → results/evaluation_report.json")
    print(f"\n  Launch dashboard:")
    print(f"      streamlit run dashboard.py")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
