# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - MLflow Model Registry & Version Management
# MAGIC 
# MAGIC ## Objectives
# MAGIC - Register best models to MLflow Model Registry
# MAGIC - Manage model versions and lifecycle
# MAGIC - Transition models through stages (Staging → Production)

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from datetime import datetime

print("Libraries imported successfully!")
print(f"Execution Time: {datetime.now()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize MLflow Client

# COMMAND ----------

# Initialize MLflow Client
experiment_name = "/Users/your_email@example.com/fraud_detection_experiments"
mlflow.set_experiment(experiment_name)
client = MlflowClient()

print(f"MLflow Client initialized")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Top Models from Experiments

# COMMAND ----------

# Get all runs and select top models
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
runs = mlflow.search_runs(experiment_ids=[experiment_id])

print(f"Total runs: {len(runs)}")

# Sort by F1-score
top_models = runs.sort_values('metrics.val_f1', ascending=False).head(5)

print("\nTop 5 Models:")
display(top_models[['tags.model_type', 'metrics.val_f1', 'metrics.val_accuracy', 'metrics.val_roc_auc']])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select Champion and Challenger

# COMMAND ----------

# Select Champion and Challenger
champion_run = top_models.iloc[0]
challenger_run = top_models.iloc[1]

print("=" * 80)
print("SELECTED MODELS")
print("=" * 80)
print(f"\nCHAMPION:")
print(f"   Model: {champion_run['tags.model_type']}")
print(f"   F1-Score: {champion_run['metrics.val_f1']:.4f}")
print(f"   Accuracy: {champion_run['metrics.val_accuracy']:.4f}")
print(f"   ROC-AUC: {champion_run['metrics.val_roc_auc']:.4f}")
print(f"   Run ID: {champion_run['run_id']}")

print(f"\nCHALLENGER:")
print(f"   Model: {challenger_run['tags.model_type']}")
print(f"   F1-Score: {challenger_run['metrics.val_f1']:.4f}")
print(f"   Accuracy: {challenger_run['metrics.val_accuracy']:.4f}")
print(f"   ROC-AUC: {challenger_run['metrics.val_roc_auc']:.4f}")
print(f"   Run ID: {challenger_run['run_id']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Champion Model

# COMMAND ----------

# Register Champion Model
model_name_champion = "fraud_detection_champion"

champion_version = mlflow.register_model(
    model_uri=f"runs:/{champion_run['run_id']}/model",
    name=model_name_champion
)

print(f"Champion registered! Version: {champion_version.version}")

# COMMAND ----------

# Transition Champion to Production
client.transition_model_version_stage(
    name=model_name_champion,
    version=champion_version.version,
    stage="Production",
    archive_existing_versions=True
)

print(f"Champion transitioned to PRODUCTION!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Challenger Model

# COMMAND ----------

# Register Challenger Model  
model_name_challenger = "fraud_detection_challenger"

challenger_version = mlflow.register_model(
    model_uri=f"runs:/{challenger_run['run_id']}/model",
    name=model_name_challenger
)

print(f"Challenger registered! Version: {challenger_version.version}")

# COMMAND ----------

# Transition Challenger to Staging
client.transition_model_version_stage(
    name=model_name_challenger,
    version=challenger_version.version,
    stage="Staging"
)

print(f"Challenger transitioned to STAGING!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Loading Models from Registry

# COMMAND ----------

# Test loading models from registry
champion_model = mlflow.sklearn.load_model(f"models:/{model_name_champion}/Production")
challenger_model = mlflow.sklearn.load_model(f"models:/{model_name_challenger}/Staging")

print("Models loaded from registry successfully!")

# COMMAND ----------

print("="*80)
print("MODEL REGISTRY COMPLETE!")
print("="*80)
print(f"\nModels ready for deployment!")
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **Notebook Complete!**
# MAGIC 
# MAGIC Proceed to: `05_champion_challenger.py`
