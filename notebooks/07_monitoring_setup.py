# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Monitoring Setup with Prometheus & Grafana

# COMMAND ----------

import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

print("Libraries imported!")

# COMMAND ----------

# Metrics configuration
metrics_config = {
    'model_performance': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
    'api_performance': ['request_count', 'latency_p99', 'error_rate', 'throughput'],
    'business_metrics': ['fraud_rate', 'total_transactions'],
    'drift_metrics': ['feature_drift_psi', 'prediction_drift']
}

print("Metrics categories:", len(metrics_config))

# COMMAND ----------

# Sample metrics data
def generate_metrics(hours=24):
    timestamps = pd.date_range(end=datetime.now(), periods=hours, freq='H')
    return pd.DataFrame([{
        'timestamp': ts,
        'accuracy': np.random.normal(0.94, 0.01),
        'f1_score': np.random.normal(0.89, 0.01),
        'latency_p99': np.random.normal(45, 5),
        'fraud_rate': np.random.uniform(0.001, 0.003),
        'avg_psi': np.random.uniform(0.05, 0.15)
    } for ts in timestamps])

metrics_df = generate_metrics(48)
print(f"Generated {len(metrics_df)} hours of metrics")

# COMMAND ----------

# Alert rules
alert_rules = [
    {'name': 'High Error Rate', 'metric': 'error_rate', 'threshold': 0.01, 'severity': 'critical'},
    {'name': 'High Latency', 'metric': 'latency_p99', 'threshold': 50, 'severity': 'warning'},
    {'name': 'Model Degradation', 'metric': 'f1_score', 'threshold': 0.85, 'severity': 'critical'},
    {'name': 'Drift Detected', 'metric': 'avg_psi', 'threshold': 0.2, 'severity': 'warning'}
]

print(f"Alert rules configured: {len(alert_rules)}")

# COMMAND ----------

# Grafana dashboard config
dashboard = {
    'title': 'Fraud Detection - Production Monitoring',
    'panels': [
        {'id': 1, 'title': 'Model Accuracy', 'type': 'graph'},
        {'id': 2, 'title': 'API Latency (p99)', 'type': 'graph'},
        {'id': 3, 'title': 'Fraud Rate', 'type': 'stat'},
        {'id': 4, 'title': 'Request Rate', 'type': 'graph'},
        {'id': 5, 'title': 'Feature Drift (PSI)', 'type': 'heatmap'}
    ]
}

print(f"Dashboard panels: {len(dashboard['panels'])}")

# COMMAND ----------

# Save configuration
dashboard_path = '/dbfs/FileStore/fraud_detection/monitoring/'
os.makedirs(dashboard_path, exist_ok=True)

with open(f'{dashboard_path}grafana_dashboard.json', 'w') as f:
    json.dump(dashboard, f, indent=2)

print(f"Saved to: {dashboard_path}")

# COMMAND ----------

# Prometheus config
prometheus_config = """
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'fraud-detection-api'
    static_configs:
      - targets: ['fraud-api:8000']
"""

with open(f'{dashboard_path}prometheus.yml', 'w') as f:
    f.write(prometheus_config)

print("Prometheus config saved")

# COMMAND ----------

print("="*80)
print("MONITORING SETUP COMPLETE!")
print("="*80)
print("All 7 notebooks complete!")
print("="*80)
