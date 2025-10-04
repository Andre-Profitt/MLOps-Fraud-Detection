# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Drift Detection & Monitoring

# COMMAND ----------

import pandas as pd
import numpy as np
from scipy import stats
import pickle

print("Libraries imported!")

# COMMAND ----------

# Load baseline
with open('/dbfs/FileStore/fraud_detection/prepared/original_train.pkl', 'rb') as f:
    X_train_baseline, y_train_baseline = pickle.load(f)
    
with open('/dbfs/FileStore/fraud_detection/prepared/original_test.pkl', 'rb') as f:
    X_test, y_test = pickle.load(f)

print(f"Baseline: {X_train_baseline.shape}")

# COMMAND ----------

def calculate_psi(baseline, current, bins=10):
    """Calculate Population Stability Index"""
    min_val = min(baseline.min(), current.min())
    max_val = max(baseline.max(), current.max())
    breakpoints = np.linspace(min_val, max_val, bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    baseline_dist = np.histogram(baseline, bins=breakpoints)[0]
    current_dist = np.histogram(current, bins=breakpoints)[0]
    
    baseline_pct = baseline_dist / len(baseline)
    current_pct = current_dist / len(current)
    
    baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
    current_pct = np.where(current_pct == 0, 0.0001, current_pct)
    
    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return psi

# COMMAND ----------

# Create current data with drift
X_current = X_test.copy()
X_current['Amount'] = X_current['Amount'] * 1.2
X_current['V1'] = X_current['V1'] + 0.5

print("Drift added to Amount and V1")

# COMMAND ----------

# Detect drift
drift_results = []
for col in X_train_baseline.columns:
    psi = calculate_psi(X_train_baseline[col].values, X_current[col].values)
    ks_stat, p_value = stats.ks_2samp(X_train_baseline[col].values, X_current[col].values)
    drift_results.append({
        'feature': col,
        'psi': psi,
        'ks_p_value': p_value,
        'drift': psi >= 0.2 or p_value < 0.05
    })

drift_df = pd.DataFrame(drift_results)
drifted = drift_df[drift_df['drift'] == True]

print(f"Drifted features: {len(drifted)}")
display(drifted)

# COMMAND ----------

# Retraining decision
should_retrain = len(drifted) > 5 or (drift_df['psi'] > 0.25).any()
print(f"Should retrain: {should_retrain}")

# COMMAND ----------

print("Drift Detection Complete!")
print("Proceed to: 07_monitoring_setup.py")
