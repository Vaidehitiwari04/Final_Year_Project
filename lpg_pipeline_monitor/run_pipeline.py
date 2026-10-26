"""
run_pipeline.py — One-click runner for Edge AI Classical Machine Learning Pipeline.
Runs data preparation, model training, multi-seed evaluation, stress tests, and artifact export.

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

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def run_step(name: str, script: str):
    print(f"\n{'=' * 60}")
    print(f"  STEP: {name}")
    print(f"{'=' * 60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], check=True)
    elapsed = time.time() - t0
    print(f"\n  ✅ {name} complete in {elapsed:.1f} s")
    return elapsed


def main():
    print("\n🔥 LPG Pipeline Monitoring System — Edge AI Classical ML Pipeline")
    print("=" * 60)
    print(f"  Working directory: {os.getcwd()}")

    total = 0
    total += run_step("Full ML Training & Evaluation Suite", "run_full_evaluation.py")

    print(f"\n{'=' * 60}")
    print(f"  🎉 ALL STEPS COMPLETE in {total:.1f} seconds")
    print(f"  📁 Trained models → models/best_model.joblib, models/danger_model.joblib")
    print(f"  📁 Results        → results/summary.csv, results/summary.md")
    print(f"\n  Launch dashboard:")
    print(f"      streamlit run dashboard.py")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
