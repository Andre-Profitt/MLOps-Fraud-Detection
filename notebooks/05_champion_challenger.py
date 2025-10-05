# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Champion/Challenger A/B Testing Deployment
# MAGIC 
# MAGIC ## Objectives
# MAGIC - Set up A/B testing framework
# MAGIC - Compare Champion vs Challenger performance  
# MAGIC - Implement traffic splitting (90/10)
# MAGIC - Evaluate which model performs better
# MAGIC - Automate model promotion logic

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pickle
import random
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")
print(f"Execution Time: {datetime.now()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Models from Registry

# COMMAND ----------

# Load Champion Model (Production)
champion_model = mlflow.sklearn.load_model("models:/fraud_detection_champion/Production")
print("Champion model loaded from Production")

# Load Challenger Model (Staging)
challenger_model = mlflow.sklearn.load_model("models:/fraud_detection_challenger/Staging")
print("Challenger model loaded from Staging")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Test Data

# COMMAND ----------

# Load test data
with open('/dbfs/FileStore/fraud_detection/prepared/original_test.pkl', 'rb') as f:
    X_test, y_test = pickle.load(f)

print(f"Test data loaded: {X_test.shape}")
print(f"Fraud cases in test: {y_test.sum()}")
print(f"Normal cases in test: {(y_test == 0).sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## A/B Testing Simulation

# COMMAND ----------

# A/B Testing Configuration
TRAFFIC_SPLIT = 0.10  # 10% to Challenger, 90% to Champion
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def assign_model(n_samples, challenger_percentage=0.10):
    """Randomly assign transactions to Champion or Challenger"""
    assignments = np.random.choice(
        ['champion', 'challenger'],
        size=n_samples,
        p=[1-challenger_percentage, challenger_percentage]
    )
    return assignments

# Assign models
model_assignments = assign_model(len(X_test), TRAFFIC_SPLIT)

print(f"Total test transactions: {len(X_test):,}")
print(f"Assigned to Champion: {(model_assignments == 'champion').sum():,} ({(model_assignments == 'champion').sum()/len(X_test)*100:.1f}%)")
print(f"Assigned to Challenger: {(model_assignments == 'challenger').sum():,} ({(model_assignments == 'challenger').sum()/len(X_test)*100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run A/B Test

# COMMAND ----------

# Make predictions based on assignment
predictions = np.zeros(len(X_test))
probabilities = np.zeros(len(X_test))

# Champion predictions
champion_mask = model_assignments == 'champion'
predictions[champion_mask] = champion_model.predict(X_test[champion_mask])
probabilities[champion_mask] = champion_model.predict_proba(X_test[champion_mask])[:, 1]

# Challenger predictions  
challenger_mask = model_assignments == 'challenger'
predictions[challenger_mask] = challenger_model.predict(X_test[challenger_mask])
probabilities[challenger_mask] = challenger_model.predict_proba(X_test[challenger_mask])[:, 1]

print("A/B test predictions completed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate Individual Model Performance

# COMMAND ----------

def evaluate_model(y_true, y_pred, y_proba, model_name):
    """Calculate comprehensive metrics for a model"""
    metrics = {
        'Model': model_name,
        'Samples': len(y_true),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_proba)
    }
    return metrics

# Evaluate Champion on its assigned traffic
champion_metrics = evaluate_model(
    y_test[champion_mask],
    predictions[champion_mask],
    probabilities[champion_mask],
    'Champion'
)

# Evaluate Challenger on its assigned traffic
challenger_metrics = evaluate_model(
    y_test[challenger_mask],
    predictions[challenger_mask],
    probabilities[challenger_mask],
    'Challenger'
)

# Display results
results_df = pd.DataFrame([champion_metrics, challenger_metrics])

print("=" * 80)
print("A/B TEST RESULTS - INDIVIDUAL MODEL PERFORMANCE")
print("=" * 80)
display(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Head-to-Head Comparison (Full Test Set)

# COMMAND ----------

# For fair comparison, evaluate both on FULL test set
champion_pred_full = champion_model.predict(X_test)
champion_proba_full = champion_model.predict_proba(X_test)[:, 1]

challenger_pred_full = challenger_model.predict(X_test)
challenger_proba_full = challenger_model.predict_proba(X_test)[:, 1]

# Evaluate both
champion_full = evaluate_model(y_test, champion_pred_full, champion_proba_full, 'Champion (Full Test)')
challenger_full = evaluate_model(y_test, challenger_pred_full, challenger_proba_full, 'Challenger (Full Test)')

comparison_df = pd.DataFrame([champion_full, challenger_full])

print("=" * 80)
print("HEAD-TO-HEAD COMPARISON (FULL TEST SET)")
print("=" * 80)
display(comparison_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Statistical Comparison

# COMMAND ----------

# Calculate differences
f1_champion = champion_full['F1-Score']
f1_challenger = challenger_full['F1-Score']
f1_diff = f1_challenger - f1_champion

print("=" * 80)
print("STATISTICAL COMPARISON")
print("=" * 80)
print(f"\nChampion F1-Score: {f1_champion:.4f}")
print(f"Challenger F1-Score: {f1_challenger:.4f}")
print(f"Difference: {f1_diff:.4f} ({f1_diff/f1_champion*100:.2f}%)")

# Determine winner
if f1_challenger > f1_champion:
    print(f"\nCHALLENGER IS BETTER by {f1_diff:.4f}!")
    winner = "Challenger"
else:
    print(f"\nCHAMPION IS BETTER by {abs(f1_diff):.4f}!")
    winner = "Champion"

# COMMAND ----------

# MAGIC %md
# MAGIC ## False Positive Reduction Analysis

# COMMAND ----------

# Champion confusion matrix
cm_champion = confusion_matrix(y_test, champion_pred_full)
tn_c, fp_c, fn_c, tp_c = cm_champion.ravel()

# Challenger confusion matrix
cm_challenger = confusion_matrix(y_test, challenger_pred_full)
tn_ch, fp_ch, fn_ch, tp_ch = cm_challenger.ravel()

# Calculate false positive rate
fpr_champion = fp_c / (fp_c + tn_c)
fpr_challenger = fp_ch / (fp_ch + tn_ch)
fp_reduction = (fpr_champion - fpr_challenger) / fpr_champion * 100

print("=" * 80)
print("FALSE POSITIVE ANALYSIS")
print("=" * 80)
print(f"\nChampion:")
print(f"  False Positives: {fp_c:,}")
print(f"  False Positive Rate: {fpr_champion:.4f}")

print(f"\nChallenger:")
print(f"  False Positives: {fp_ch:,}")
print(f"  False Positive Rate: {fpr_challenger:.4f}")

print(f"\nFalse Positive Reduction: {fp_reduction:.2f}%")

if fp_reduction > 0:
    print(f"Challenger reduces false positives by {fp_reduction:.1f}%!")
else:
    print(f"Champion has {abs(fp_reduction):.1f}% fewer false positives")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Automated Model Promotion Logic

# COMMAND ----------

# Define promotion criteria
PROMOTION_THRESHOLD = 0.02  # Challenger must be 2% better
MIN_F1_SCORE = 0.85  # Minimum acceptable F1

def should_promote_challenger(champion_f1, challenger_f1, threshold=PROMOTION_THRESHOLD, min_f1=MIN_F1_SCORE):
    """Determine if challenger should be promoted to production"""
    # Check if challenger meets minimum threshold
    if challenger_f1 < min_f1:
        return False, f"Challenger F1 ({challenger_f1:.4f}) below minimum ({min_f1})"
    
    # Check if challenger is significantly better
    improvement = (challenger_f1 - champion_f1) / champion_f1
    
    if improvement > threshold:
        return True, f"Challenger improves F1 by {improvement*100:.2f}% (threshold: {threshold*100}%)"
    else:
        return False, f"Improvement ({improvement*100:.2f}%) below threshold ({threshold*100}%)"

# Execute promotion logic
should_promote, reason = should_promote_challenger(f1_champion, f1_challenger)

print("=" * 80)
print("AUTOMATED PROMOTION DECISION")
print("=" * 80)
print(f"\nDecision: {'PROMOTE CHALLENGER' if should_promote else 'KEEP CHAMPION'}")
print(f"Reason: {reason}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute Promotion (if applicable)

# COMMAND ----------

if should_promote:
    print("Promoting Challenger to Production...")
    
    client = MlflowClient()
    
    # Get challenger version
    challenger_versions = client.get_latest_versions("fraud_detection_challenger", stages=["Staging"])
    challenger_version = challenger_versions[0].version
    
    # Transition challenger to Production
    client.transition_model_version_stage(
        name="fraud_detection_challenger",
        version=challenger_version,
        stage="Production",
        archive_existing_versions=False
    )
    
    # Archive old champion
    champion_versions = client.get_latest_versions("fraud_detection_champion", stages=["Production"])
    if champion_versions:
        old_champion_version = champion_versions[0].version
        client.transition_model_version_stage(
            name="fraud_detection_champion",
            version=old_champion_version,
            stage="Archived"
        )
    
    print("Challenger promoted to Production!")
    print("Old Champion archived")
    
else:
    print("Champion remains in Production")
    print("Continue monitoring Challenger performance")

# COMMAND ----------

# MAGIC %md
# MAGIC ## A/B Test Summary Report

# COMMAND ----------

# Create summary report
import json
import os

summary = {
    'Test Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'Traffic Split': f"{int((1-TRAFFIC_SPLIT)*100)}% Champion / {int(TRAFFIC_SPLIT*100)}% Challenger",
    'Test Samples': len(X_test),
    'Champion Samples': int((model_assignments == 'champion').sum()),
    'Challenger Samples': int((model_assignments == 'challenger').sum()),
    'Winner': winner,
    'Champion F1': f"{f1_champion:.4f}",
    'Challenger F1': f"{f1_challenger:.4f}",
    'F1 Improvement': f"{f1_diff:.4f} ({f1_diff/f1_champion*100:.2f}%)",
    'FP Reduction': f"{fp_reduction:.2f}%",
    'Promotion': 'Yes' if should_promote else 'No',
    'Reason': reason
}

print("=" * 80)
print("A/B TEST SUMMARY REPORT")
print("=" * 80)
for key, value in summary.items():
    print(f"{key:<25} {value}")
print("=" * 80)

# Save report
report_path = '/dbfs/FileStore/fraud_detection/reports/'
os.makedirs(report_path, exist_ok=True)

with open(f'{report_path}ab_test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Report saved to: {report_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualization

# COMMAND ----------

# Create comparison visualization
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Metrics comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
champion_values = [champion_full[m] for m in metrics]
challenger_values = [challenger_full[m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

axes[0].bar(x - width/2, champion_values, width, label='Champion', color='#2ecc71', alpha=0.8)
axes[0].bar(x + width/2, challenger_values, width, label='Challenger', color='#3498db', alpha=0.8)
axes[0].set_ylabel('Score')
axes[0].set_title('Model Performance Comparison', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim([0, 1])

# Confusion matrix comparison
categories = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
champion_cm = [tn_c, fp_c, fn_c, tp_c]
challenger_cm = [tn_ch, fp_ch, fn_ch, tp_ch]

x2 = np.arange(len(categories))
axes[1].bar(x2 - width/2, champion_cm, width, label='Champion', color='#2ecc71', alpha=0.8)
axes[1].bar(x2 + width/2, challenger_cm, width, label='Challenger', color='#3498db', alpha=0.8)
axes[1].set_ylabel('Count')
axes[1].set_title('Confusion Matrix Comparison', fontweight='bold')
axes[1].set_xticks(x2)
axes[1].set_xticklabels(categories, rotation=45, ha='right')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/dbfs/FileStore/fraud_detection/visualizations/ab_test_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC A/B Testing Complete - Models compared on real test data - Promotion decision automated
# MAGIC 
# MAGIC Next: `06_drift_detection.py`

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **Notebook Complete!**
# MAGIC 
# MAGIC Proceed to: `06_drift_detection.py`
