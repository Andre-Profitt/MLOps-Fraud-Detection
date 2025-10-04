# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Champion/Challenger A/B Testing

# COMMAND ----------

import mlflow.sklearn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import random

print("Libraries imported!")

# COMMAND ----------

# Load models
champion_model = mlflow.sklearn.load_model("models:/fraud_detection_champion/Production")
challenger_model = mlflow.sklearn.load_model("models:/fraud_detection_challenger/Staging")
print("Models loaded")

# COMMAND ----------

# Load test data
with open('/dbfs/FileStore/fraud_detection/prepared/original_test.pkl', 'rb') as f:
    X_test, y_test = pickle.load(f)
print(f"Test data: {X_test.shape}")

# COMMAND ----------

# A/B test - 90% Champion, 10% Challenger
TRAFFIC_SPLIT = 0.10
assignments = np.random.choice(['champion', 'challenger'], size=len(X_test), p=[0.9, 0.1])

print(f"Champion: {(assignments=='champion').sum()}")
print(f"Challenger: {(assignments=='challenger').sum()}")

# COMMAND ----------

# Make predictions
predictions = np.zeros(len(X_test))
champion_mask = assignments == 'champion'
challenger_mask = assignments == 'challenger'

predictions[champion_mask] = champion_model.predict(X_test[champion_mask])
predictions[challenger_mask] = challenger_model.predict(X_test[challenger_mask])

# COMMAND ----------

# Evaluate both models on full test set
champ_pred = champion_model.predict(X_test)
chal_pred = challenger_model.predict(X_test)

print("Champion F1:", f1_score(y_test, champ_pred))
print("Challenger F1:", f1_score(y_test, chal_pred))

# COMMAND ----------

# Confusion matrices
cm_champ = confusion_matrix(y_test, champ_pred)
cm_chal = confusion_matrix(y_test, chal_pred)

tn_c, fp_c, fn_c, tp_c = cm_champ.ravel()
tn_ch, fp_ch, fn_ch, tp_ch = cm_chal.ravel()

fp_reduction = (fp_c - fp_ch) / fp_c * 100
print(f"False Positive Reduction: {fp_reduction:.2f}%")

# COMMAND ----------

print("A/B Testing Complete!")
print("Proceed to: 06_drift_detection.py")
