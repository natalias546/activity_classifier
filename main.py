import os
import sys
import subprocess
import argparse
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PARQUET_WATCH = os.path.join(BASE, "data", "watch_features.parquet")
PARQUET_PHONE = os.path.join(BASE, "data", "phone_features.parquet")

def run(script, label):
    print(f"Running: {label}")
    start = time.time()
    result = subprocess.run([sys.executable, os.path.join(BASE, script)])
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\nERROR: {label} failed. Stopping pipeline.")
        sys.exit(1)
    print(f"\nDone: {label} ({elapsed / 60:.1f} min)")

def parquet_exists():
    return os.path.exists(PARQUET_WATCH) and os.path.exists(PARQUET_PHONE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Activity Classifier Pipeline")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature engineering if parquet files already exist")
    parser.add_argument("--eda", action="store_true",
                        help="Run EDA after model training")
    args = parser.parse_args()

    if args.skip_features and parquet_exists():
        print("Parquet files found — skipping feature engineering.")
    else:
        run("rf_pipeline/feature_engineering.py", "Feature Engineering")

    run("rf_pipeline/model.py", "Model Training & Evaluation")

    if args.eda:
        run("rf_pipeline/eda.py", "EDA")

    print("Pipeline completed")
