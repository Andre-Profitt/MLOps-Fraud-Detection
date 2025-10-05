# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Drift Detection & Automated Monitoring
# MAGIC 
# MAGIC ## Objectives
# MAGIC - Implement drift detection algorithms (PSI, KS Test)
# MAGIC - Monitor feature drift
# MAGIC - Monitor prediction drift
# MAGIC - Set up automated retraining triggers
# MAGIC - Create drift monitoring dashboard

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Imports

# COMMAND ----------

import pandas as pd
import numpy as np
from scipy import stats
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# For production drift detection
import sys
sys.path.append('/dbfs/FileStore/fraud_detection/src/')

print("Libraries imported successfully!")
print(f"Execution Time: {datetime.now()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Baseline and Current Data

# COMMAND ----------

# Load baseline (training data) - this is our reference distribution
with open('/dbfs/FileStore/fraud_detection/prepared/original_train.pkl', 'rb') as f:
    X_train_baseline, y_train_baseline = pickle.load(f)

# Load test data (simulating production data)
with open('/dbfs/FileStore/fraud_detection/prepared/original_test.pkl', 'rb') as f:
    X_test, y_test = pickle.load(f)

print(f"Baseline data shape: {X_train_baseline.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Features: {X_train_baseline.shape[1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Population Stability Index (PSI) Implementation

# COMMAND ----------

def calculate_psi(baseline_array, current_array, bins=10, return_components=False):
    """
    Calculate Population Stability Index (PSI)
    
    PSI Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Small change
    - PSI >= 0.2: Significant change (requires action)
    
    Args:
        baseline_array: Reference distribution
        current_array: Current distribution
        bins: Number of bins for discretization
        return_components: Return detailed breakdown
    
    Returns:
        PSI value (float) or (PSI, components) if return_components=True
    """
    # Define bins based on baseline
    min_val = min(baseline_array.min(), current_array.min())
    max_val = max(baseline_array.max(), current_array.max())
    
    breakpoints = np.linspace(min_val, max_val, bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    # Calculate distributions
    baseline_dist = np.histogram(baseline_array, bins=breakpoints)[0]
    current_dist = np.histogram(current_array, bins=breakpoints)[0]
    
    # Convert to percentages
    baseline_pct = baseline_dist / len(baseline_array)
    current_pct = current_dist / len(current_array)
    
    # Avoid division by zero
    baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
    current_pct = np.where(current_pct == 0, 0.0001, current_pct)
    
    # Calculate PSI for each bin
    psi_values = (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
    psi = np.sum(psi_values)
    
    if return_components:
        return psi, {
            'bins': breakpoints.tolist(),
            'baseline_pct': baseline_pct.tolist(),
            'current_pct': current_pct.tolist(),
            'psi_per_bin': psi_values.tolist()
        }
    
    return psi

# Test PSI calculation
sample_feature = 'Amount'
psi_value = calculate_psi(
    X_train_baseline[sample_feature].values,
    X_test[sample_feature].values
)
print(f"PSI for {sample_feature}: {psi_value:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Kolmogorov-Smirnov (KS) Test Implementation

# COMMAND ----------

def calculate_ks_statistic(baseline_array, current_array):
    """
    Calculate Kolmogorov-Smirnov test statistic
    
    KS Test Interpretation:
    - p-value < 0.05: Distributions are significantly different
    - p-value >= 0.05: No significant difference
    
    Args:
        baseline_array: Reference distribution
        current_array: Current distribution
    
    Returns:
        Tuple of (KS statistic, p-value)
    """
    ks_statistic, p_value = stats.ks_2samp(baseline_array, current_array)
    return ks_statistic, p_value

# Test KS calculation
ks_stat, p_value = calculate_ks_statistic(
    X_train_baseline[sample_feature].values,
    X_test[sample_feature].values
)
print(f"KS Statistic for {sample_feature}: {ks_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Drift detected: {p_value < 0.05}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Calculate Drift for All Features

# COMMAND ----------

def calculate_feature_drift(baseline_data, current_data, psi_threshold=0.2, ks_threshold=0.05):
    """
    Calculate drift metrics for all features
    """
    drift_results = []
    
    for col in baseline_data.columns:
        baseline_values = baseline_data[col].values
        current_values = current_data[col].values
        
        # Calculate PSI
        psi = calculate_psi(baseline_values, current_values)
        
        # Calculate KS test
        ks_stat, p_value = calculate_ks_statistic(baseline_values, current_values)
        
        # Determine drift
        psi_drift = psi >= psi_threshold
        ks_drift = p_value < ks_threshold
        overall_drift = psi_drift or ks_drift
        
        drift_results.append({
            'feature': col,
            'psi': psi,
            'psi_threshold': psi_threshold,
            'psi_drift': psi_drift,
            'ks_statistic': ks_stat,
            'ks_p_value': p_value,
            'ks_threshold': ks_threshold,
            'ks_drift': ks_drift,
            'drift_detected': overall_drift,
            'baseline_mean': baseline_values.mean(),
            'current_mean': current_values.mean(),
            'mean_shift': current_values.mean() - baseline_values.mean(),
            'baseline_std': baseline_values.std(),
            'current_std': current_values.std()
        })
    
    return pd.DataFrame(drift_results)

# Calculate drift for all features
drift_df = calculate_feature_drift(X_train_baseline, X_test)

# Display results
print("=" * 100)
print("FEATURE DRIFT ANALYSIS")
print("=" * 100)

# Sort by PSI
drift_df_sorted = drift_df.sort_values('psi', ascending=False)
print("\nTop 10 Features by PSI:")
display(drift_df_sorted.head(10)[['feature', 'psi', 'psi_drift', 'ks_p_value', 'drift_detected']])

# Features with drift
drifted_features = drift_df[drift_df['drift_detected'] == True]
print(f"\nFeatures with drift detected: {len(drifted_features)} / {len(drift_df)}")

if len(drifted_features) > 0:
    print("\nDrifted features:")
    display(drifted_features[['feature', 'psi', 'ks_p_value', 'mean_shift']])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Simulate Production Data with Drift

# COMMAND ----------

# Create synthetic production data with intentional drift
X_production = X_test.copy()

# Add drift to simulate production changes
np.random.seed(42)

# Scenario 1: Amount feature shifts (inflation, changing transaction patterns)
X_production['Amount'] = X_production['Amount'] * 1.2 + np.random.normal(0, 10, len(X_production))

# Scenario 2: V1 and V2 shift (changing fraud patterns)
X_production['V1'] = X_production['V1'] + np.random.normal(0.5, 0.1, len(X_production))
X_production['V2'] = X_production['V2'] * 0.9

# Scenario 3: Time-based features shift
if 'Hour' in X_production.columns:
    X_production['Hour'] = (X_production['Hour'] + 3) % 24

print("Synthetic drift added to production data:")
print("- Amount: 20% increase + noise")
print("- V1: Mean shift of +0.5")
print("- V2: 10% scale reduction")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Detect Drift in Production Data

# COMMAND ----------

# Calculate drift for production data
drift_prod = calculate_feature_drift(X_train_baseline, X_production)

# Display results
print("=" * 100)
print("PRODUCTION DATA DRIFT ANALYSIS")
print("=" * 100)

# Features with drift
drifted_prod = drift_prod[drift_prod['drift_detected'] == True]
print(f"\nDrifted features in production: {len(drifted_prod)} / {len(drift_prod)}")

if len(drifted_prod) > 0:
    print("\nDrifted features (Production):")
    display(drifted_prod.sort_values('psi', ascending=False)[['feature', 'psi', 'ks_p_value', 'mean_shift']])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Prediction Drift Detection

# COMMAND ----------

import mlflow.sklearn

# Load model for predictions
model = mlflow.sklearn.load_model("models:/fraud_detection_champion/Production")

# Make predictions on baseline and production data
baseline_predictions = model.predict(X_train_baseline)
baseline_probabilities = model.predict_proba(X_train_baseline)[:, 1]

production_predictions = model.predict(X_production)
production_probabilities = model.predict_proba(X_production)[:, 1]

# Calculate prediction drift
baseline_fraud_rate = baseline_predictions.mean()
production_fraud_rate = production_predictions.mean()

print("=" * 100)
print("PREDICTION DRIFT ANALYSIS")
print("=" * 100)
print(f"Baseline fraud rate: {baseline_fraud_rate:.4f} ({baseline_fraud_rate*100:.2f}%)")
print(f"Production fraud rate: {production_fraud_rate:.4f} ({production_fraud_rate*100:.2f}%)")
print(f"Absolute change: {production_fraud_rate - baseline_fraud_rate:.4f}")
print(f"Relative change: {((production_fraud_rate - baseline_fraud_rate) / baseline_fraud_rate * 100):.2f}%")

# Test for statistical significance
chi2_stat, chi2_p = stats.chisquare(
    [production_predictions.sum(), len(production_predictions) - production_predictions.sum()],
    [baseline_predictions.sum() * len(production_predictions) / len(baseline_predictions),
     (len(baseline_predictions) - baseline_predictions.sum()) * len(production_predictions) / len(baseline_predictions)]
)

print(f"\nChi-square test p-value: {chi2_p:.4f}")
print(f"Significant drift in predictions: {chi2_p < 0.05}")

# Probability distribution drift
prob_psi = calculate_psi(baseline_probabilities, production_probabilities)
prob_ks_stat, prob_ks_p = calculate_ks_statistic(baseline_probabilities, production_probabilities)

print(f"\nProbability distribution drift:")
print(f"  PSI: {prob_psi:.4f}")
print(f"  KS p-value: {prob_ks_p:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Automated Retraining Decision

# COMMAND ----------

def should_trigger_retraining(drift_df, prediction_drift_metrics, 
                              max_drifted_features=5,
                              psi_critical_threshold=0.25,
                              prediction_shift_threshold=0.05):
    """
    Determine if model retraining should be triggered
    
    Retraining triggered if:
    1. Too many features have drifted (> max_drifted_features)
    2. Any feature has critical PSI (> psi_critical_threshold)
    3. Prediction distribution shifts significantly
    """
    reasons = []
    should_retrain = False
    
    # Check feature drift
    drifted_count = drift_df['drift_detected'].sum()
    if drifted_count > max_drifted_features:
        reasons.append(f"Too many features drifted: {drifted_count} > {max_drifted_features}")
        should_retrain = True
    
    # Check for critical drift
    critical_features = drift_df[drift_df['psi'] > psi_critical_threshold]
    if len(critical_features) > 0:
        reasons.append(f"Critical drift in: {', '.join(critical_features['feature'].head(3).values)}")
        should_retrain = True
    
    # Check prediction drift
    if abs(prediction_drift_metrics['rate_change']) > prediction_shift_threshold:
        reasons.append(f"Prediction rate shifted by {prediction_drift_metrics['rate_change']:.2%}")
        should_retrain = True
    
    if prediction_drift_metrics['probability_psi'] > 0.2:
        reasons.append(f"Probability distribution PSI: {prediction_drift_metrics['probability_psi']:.3f}")
        should_retrain = True
    
    return should_retrain, reasons

# Prepare metrics
prediction_drift_metrics = {
    'rate_change': production_fraud_rate - baseline_fraud_rate,
    'probability_psi': prob_psi,
    'chi2_p_value': chi2_p
}

# Make retraining decision
should_retrain, reasons = should_trigger_retraining(drift_prod, prediction_drift_metrics)

print("=" * 100)
print("AUTOMATED RETRAINING DECISION")
print("=" * 100)
print(f"Decision: {'TRIGGER RETRAINING' if should_retrain else 'CONTINUE MONITORING'}")

if reasons:
    print("\nReasons:")
    for reason in reasons:
        print(f"  - {reason}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Drift Visualization

# COMMAND ----------

# Create comprehensive drift visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. PSI by feature
ax = axes[0, 0]
top_psi = drift_prod.nlargest(15, 'psi')
colors = ['red' if x >= 0.2 else 'orange' if x >= 0.1 else 'green' for x in top_psi['psi']]
ax.barh(range(len(top_psi)), top_psi['psi'], color=colors)
ax.set_yticks(range(len(top_psi)))
ax.set_yticklabels(top_psi['feature'])
ax.set_xlabel('PSI Value')
ax.set_title('Top 15 Features by PSI', fontweight='bold')
ax.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, label='Small change')
ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.5, label='Significant change')
ax.legend()
ax.invert_yaxis()

# 2. KS Test p-values
ax = axes[0, 1]
top_ks = drift_prod.nsmallest(15, 'ks_p_value')
colors = ['red' if x < 0.05 else 'green' for x in top_ks['ks_p_value']]
ax.barh(range(len(top_ks)), 1 - top_ks['ks_p_value'], color=colors)
ax.set_yticks(range(len(top_ks)))
ax.set_yticklabels(top_ks['feature'])
ax.set_xlabel('1 - p-value (higher = more drift)')
ax.set_title('Top 15 Features by KS Test', fontweight='bold')
ax.axvline(x=0.95, color='red', linestyle='--', alpha=0.5, label='p < 0.05')
ax.legend()
ax.invert_yaxis()

# 3. Drift summary
ax = axes[0, 2]
drift_summary = pd.Series({
    'No Drift': (~drift_prod['drift_detected']).sum(),
    'PSI Drift': drift_prod['psi_drift'].sum(),
    'KS Drift': drift_prod['ks_drift'].sum(),
    'Both': (drift_prod['psi_drift'] & drift_prod['ks_drift']).sum()
})
colors = ['green', 'orange', 'yellow', 'red']
wedges, texts, autotexts = ax.pie(drift_summary, labels=drift_summary.index, autopct='%1.0f',
                                   colors=colors, startangle=90)
ax.set_title('Feature Drift Summary', fontweight='bold')

# 4. Mean shift distribution
ax = axes[1, 0]
ax.hist(drift_prod['mean_shift'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Mean Shift')
ax.set_ylabel('Number of Features')
ax.set_title('Distribution of Mean Shifts', fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)

# 5. Prediction distribution comparison
ax = axes[1, 1]
bins = np.linspace(0, 1, 30)
ax.hist(baseline_probabilities, bins=bins, alpha=0.5, label='Baseline', color='blue', density=True)
ax.hist(production_probabilities, bins=bins, alpha=0.5, label='Production', color='red', density=True)
ax.set_xlabel('Fraud Probability')
ax.set_ylabel('Density')
ax.set_title('Prediction Probability Distribution', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 6. Time series drift (simulated)
ax = axes[1, 2]
days = 30
dates = pd.date_range(end=datetime.now(), periods=days)
simulated_psi = np.random.normal(0.15, 0.05, days)
simulated_psi = np.cumsum(simulated_psi - 0.145)  # Add trend
simulated_psi = np.maximum(simulated_psi, 0)
ax.plot(dates, simulated_psi, marker='o', markersize=4, linewidth=2)
ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Small change')
ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Significant change')
ax.set_xlabel('Date')
ax.set_ylabel('Average PSI')
ax.set_title('Drift Trend Over Time (Simulated)', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/dbfs/FileStore/fraud_detection/visualizations/drift_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Save Drift Report

# COMMAND ----------

import os

# Create comprehensive drift report
drift_report = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'baseline_samples': len(X_train_baseline),
    'production_samples': len(X_production),
    'feature_drift': {
        'total_features': len(drift_prod),
        'drifted_features': int(drift_prod['drift_detected'].sum()),
        'psi_drifted': int(drift_prod['psi_drift'].sum()),
        'ks_drifted': int(drift_prod['ks_drift'].sum()),
        'max_psi': float(drift_prod['psi'].max()),
        'mean_psi': float(drift_prod['psi'].mean()),
        'critical_features': drift_prod[drift_prod['psi'] > 0.25]['feature'].tolist()
    },
    'prediction_drift': {
        'baseline_fraud_rate': float(baseline_fraud_rate),
        'production_fraud_rate': float(production_fraud_rate),
        'rate_change': float(production_fraud_rate - baseline_fraud_rate),
        'probability_psi': float(prob_psi),
        'chi2_p_value': float(chi2_p),
        'significant_drift': chi2_p < 0.05
    },
    'retraining': {
        'should_retrain': should_retrain,
        'reasons': reasons
    },
    'top_drifted_features': drift_prod.nlargest(5, 'psi')[['feature', 'psi', 'mean_shift']].to_dict('records')
}

# Save report
report_path = '/dbfs/FileStore/fraud_detection/drift_reports/'
os.makedirs(report_path, exist_ok=True)

report_filename = f'{report_path}drift_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(report_filename, 'w') as f:
    json.dump(drift_report, f, indent=2)

print(f"Drift report saved to: {report_filename}")

# Display summary
print("\n" + "=" * 100)
print("DRIFT REPORT SUMMARY")
print("=" * 100)
print(f"Features with drift: {drift_report['feature_drift']['drifted_features']} / {drift_report['feature_drift']['total_features']}")
print(f"Maximum PSI: {drift_report['feature_drift']['max_psi']:.3f}")
print(f"Prediction rate change: {drift_report['prediction_drift']['rate_change']:.4f}")
print(f"Retraining recommended: {drift_report['retraining']['should_retrain']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Set Up Automated Monitoring

# COMMAND ----------

# Create monitoring schedule configuration
monitoring_config = {
    'enabled': True,
    'schedule': {
        'frequency': 'hourly',
        'start_time': '00:00',
        'timezone': 'UTC'
    },
    'thresholds': {
        'psi_warning': 0.1,
        'psi_critical': 0.2,
        'ks_p_value': 0.05,
        'max_drifted_features': 5,
        'prediction_shift_threshold': 0.05
    },
    'alerts': {
        'email': ['data-team@example.com'],
        'slack_webhook': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
        'pagerduty': False
    },
    'actions': {
        'auto_retrain': False,  # Set to True for automatic retraining
        'notify_on_drift': True,
        'log_to_mlflow': True
    }
}

# Save monitoring config
config_path = '/dbfs/FileStore/fraud_detection/configs/'
os.makedirs(config_path, exist_ok=True)

with open(f'{config_path}drift_monitoring_config.json', 'w') as f:
    json.dump(monitoring_config, f, indent=2)

print("Monitoring configuration saved")
print(f"Monitoring frequency: {monitoring_config['schedule']['frequency']}")
print(f"Auto-retrain enabled: {monitoring_config['actions']['auto_retrain']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC Drift Detection Complete!
# MAGIC 
# MAGIC Key Findings:
# MAGIC - Feature drift detected in production data
# MAGIC - PSI and KS tests implemented
# MAGIC - Automated retraining decision logic in place
# MAGIC - Monitoring configured for continuous drift detection
# MAGIC 
# MAGIC Next: `07_monitoring_setup.py`

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **Notebook Complete!** ✓
# MAGIC 
# MAGIC Proceed to: `07_monitoring_setup.py`
