# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Model Training with MLflow Experiment Tracking
# MAGIC 
# MAGIC ## Objectives
# MAGIC - Set up MLflow experiment tracking
# MAGIC - Train 20+ model iterations with different algorithms
# MAGIC - Log hyperparameters, metrics, and artifacts
# MAGIC - Compare model performance
# MAGIC - Select best model for deployment

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Imports

# COMMAND ----------

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")
print(f"MLflow version: {mlflow.__version__}")
print(f"Execution Time: {datetime.now()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Prepared Data

# COMMAND ----------

# Load data from previous notebook
data_path = '/dbfs/FileStore/fraud_detection/prepared/'

# Load original (imbalanced) data
with open(f'{data_path}original_train.pkl', 'rb') as f:
    X_train_orig, y_train_orig = pickle.load(f)

with open(f'{data_path}original_val.pkl', 'rb') as f:
    X_val, y_val = pickle.load(f)

with open(f'{data_path}original_test.pkl', 'rb') as f:
    X_test, y_test = pickle.load(f)

# Load SMOTE data
with open(f'{data_path}smote_train.pkl', 'rb') as f:
    X_train_smote, y_train_smote = pickle.load(f)

# Load class weights
with open(f'{data_path}class_weights.pkl', 'rb') as f:
    class_weights = pickle.load(f)

# Load feature names
with open(f'{data_path}feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print(f"Data loaded successfully!")
print(f"Train (original): {X_train_orig.shape}")
print(f"Train (SMOTE): {X_train_smote.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test: {X_test.shape}")
print(f"Features: {len(feature_names)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. MLflow Experiment Setup

# COMMAND ----------

# Set experiment name
experiment_name = "/Users/your_email@example.com/fraud_detection_experiments"

# Create or get existing experiment
try:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"Created new experiment: {experiment_name}")
except:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    print(f"Using existing experiment: {experiment_name}")

mlflow.set_experiment(experiment_name)
print(f"Experiment ID: {experiment_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Evaluation Helper Functions

# COMMAND ----------

def evaluate_model(model, X, y, dataset_name="Dataset"):
    """
    Comprehensive model evaluation
    """
    # Predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    metrics = {
        f'{dataset_name}_accuracy': accuracy_score(y, y_pred),
        f'{dataset_name}_precision': precision_score(y, y_pred),
        f'{dataset_name}_recall': recall_score(y, y_pred),
        f'{dataset_name}_f1': f1_score(y, y_pred),
        f'{dataset_name}_roc_auc': roc_auc_score(y, y_pred_proba),
        f'{dataset_name}_avg_precision': average_precision_score(y, y_pred_proba)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics[f'{dataset_name}_true_negatives'] = int(tn)
    metrics[f'{dataset_name}_false_positives'] = int(fp)
    metrics[f'{dataset_name}_false_negatives'] = int(fn)
    metrics[f'{dataset_name}_true_positives'] = int(tp)
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics[f'{dataset_name}_specificity'] = specificity
    
    return metrics, y_pred, y_pred_proba


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """
    Plot confusion matrix
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticklabels(['Normal', 'Fraud'])
    ax.set_yticklabels(['Normal', 'Fraud'])
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve"):
    """
    Plot ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_precision_recall_curve(y_true, y_pred_proba, title="Precision-Recall Curve"):
    """
    Plot Precision-Recall curve
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

print("Helper functions defined!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Model Training Pipeline

# COMMAND ----------

def train_and_log_model(model, model_name, X_train, y_train, X_val, y_val, 
                       params={}, tags={}, use_smote=False):
    """
    Train model and log everything to MLflow
    """
    with mlflow.start_run(run_name=model_name) as run:
        
        # Log tags
        mlflow.set_tags({
            'model_type': model_name,
            'resampling': 'SMOTE' if use_smote else 'Original',
            **tags
        })
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param('n_features', X_train.shape[1])
        mlflow.log_param('n_train_samples', X_train.shape[0])
        
        # Train model
        print(f"\n{'='*80}")
        print(f"Training: {model_name}")
        print(f"{'='*80}")
        
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        mlflow.log_metric('training_time_seconds', training_time)
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on train set
        train_metrics, _, train_proba = evaluate_model(model, X_train, y_train, "train")
        
        # Evaluate on validation set
        val_metrics, val_pred, val_proba = evaluate_model(model, X_val, y_val, "val")
        
        # Log all metrics
        mlflow.log_metrics({**train_metrics, **val_metrics})
        
        # Print validation results
        print(f"\nValidation Results:")
        print(f"  Accuracy:  {val_metrics['val_accuracy']:.4f}")
        print(f"  Precision: {val_metrics['val_precision']:.4f}")
        print(f"  Recall:    {val_metrics['val_recall']:.4f}")
        print(f"  F1-Score:  {val_metrics['val_f1']:.4f}")
        print(f"  ROC-AUC:   {val_metrics['val_roc_auc']:.4f}")
        
        # Create and log confusion matrix
        cm = confusion_matrix(y_val, val_pred)
        cm_fig = plot_confusion_matrix(cm, f"{model_name} - Validation Confusion Matrix")
        mlflow.log_figure(cm_fig, f"{model_name}_confusion_matrix.png")
        plt.close()
        
        # Create and log ROC curve
        roc_fig = plot_roc_curve(y_val, val_proba, f"{model_name} - ROC Curve")
        mlflow.log_figure(roc_fig, f"{model_name}_roc_curve.png")
        plt.close()
        
        # Create and log PR curve
        pr_fig = plot_precision_recall_curve(y_val, val_proba, f"{model_name} - PR Curve")
        mlflow.log_figure(pr_fig, f"{model_name}_pr_curve.png")
        plt.close()
        
        # Log model
        if 'XGB' in model_name:
            mlflow.xgboost.log_model(model, "model")
        elif 'LGBM' in model_name or 'LightGBM' in model_name:
            mlflow.lightgbm.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
        
        # Log feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Save feature importance
            temp_path = f"/tmp/{model_name}_feature_importance.csv"
            feature_importance.to_csv(temp_path, index=False)
            mlflow.log_artifact(temp_path, "feature_importance")
            os.remove(temp_path)
            
            # Plot top 20 features
            fig, ax = plt.subplots(figsize=(10, 8))
            top_features = feature_importance.head(20)
            ax.barh(range(len(top_features)), top_features['importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Importance', fontweight='bold')
            ax.set_title(f'{model_name} - Top 20 Feature Importances', fontweight='bold')
            ax.invert_yaxis()
            plt.tight_layout()
            mlflow.log_figure(fig, f"{model_name}_feature_importance.png")
            plt.close()
        
        print(f"✓ Run logged to MLflow (Run ID: {run.info.run_id})")
        
        return model, val_metrics

print("Training pipeline ready!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Baseline Models

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Logistic Regression (Baseline)

# COMMAND ----------

# Model 1: Logistic Regression - Original Data
lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

lr_params = {
    'penalty': 'l2',
    'C': 1.0,
    'solver': 'lbfgs',
    'class_weight': 'balanced'
}

model_lr, metrics_lr = train_and_log_model(
    lr_model, 
    "Logistic_Regression_Baseline",
    X_train_orig, y_train_orig, X_val, y_val,
    params=lr_params,
    tags={'iteration': '1', 'baseline': 'true'}
)

# COMMAND ----------

# Model 2: Logistic Regression - SMOTE
lr_smote = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_smote_params = {
    'penalty': 'l2',
    'C': 1.0,
    'solver': 'lbfgs'
}

model_lr_smote, metrics_lr_smote = train_and_log_model(
    lr_smote,
    "Logistic_Regression_SMOTE",
    X_train_smote, y_train_smote, X_val, y_val,
    params=lr_smote_params,
    tags={'iteration': '2'},
    use_smote=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 Decision Tree

# COMMAND ----------

# Model 3: Decision Tree
dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=100,
    min_samples_leaf=50,
    class_weight='balanced',
    random_state=42
)

dt_params = {
    'max_depth': 10,
    'min_samples_split': 100,
    'min_samples_leaf': 50,
    'criterion': 'gini',
    'class_weight': 'balanced'
}

model_dt, metrics_dt = train_and_log_model(
    dt_model,
    "Decision_Tree",
    X_train_orig, y_train_orig, X_val, y_val,
    params=dt_params,
    tags={'iteration': '3'}
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Ensemble Models

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.1 Random Forest

# COMMAND ----------

# Model 4: Random Forest - Balanced
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=50,
    min_samples_leaf=25,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_params = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 50,
    'min_samples_leaf': 25,
    'max_features': 'sqrt',
    'class_weight': 'balanced'
}

model_rf, metrics_rf = train_and_log_model(
    rf_model,
    "Random_Forest_Balanced",
    X_train_orig, y_train_orig, X_val, y_val,
    params=rf_params,
    tags={'iteration': '4'}
)

# COMMAND ----------

# Model 5: Random Forest - SMOTE
rf_smote = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf_smote_params = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'max_features': 'sqrt'
}

model_rf_smote, metrics_rf_smote = train_and_log_model(
    rf_smote,
    "Random_Forest_SMOTE",
    X_train_smote, y_train_smote, X_val, y_val,
    params=rf_smote_params,
    tags={'iteration': '5'},
    use_smote=True
)

# COMMAND ----------

# Model 6: Random Forest - Optimized
rf_opt = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

rf_opt_params = {
    'n_estimators': 300,
    'max_depth': 25,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'class_weight': 'balanced_subsample'
}

model_rf_opt, metrics_rf_opt = train_and_log_model(
    rf_opt,
    "Random_Forest_Optimized",
    X_train_orig, y_train_orig, X_val, y_val,
    params=rf_opt_params,
    tags={'iteration': '6'}
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.2 Gradient Boosting

# COMMAND ----------

# Model 7: Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

gb_params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'subsample': 0.8
}

model_gb, metrics_gb = train_and_log_model(
    gb_model,
    "Gradient_Boosting",
    X_train_smote, y_train_smote, X_val, y_val,
    params=gb_params,
    tags={'iteration': '7'},
    use_smote=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. XGBoost Models

# COMMAND ----------

# Model 8: XGBoost - Balanced
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=(y_train_orig == 0).sum() / (y_train_orig == 1).sum(),
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

xgb_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'scale_pos_weight': float((y_train_orig == 0).sum() / (y_train_orig == 1).sum()),
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

model_xgb, metrics_xgb = train_and_log_model(
    xgb_model,
    "XGBoost_Balanced",
    X_train_orig, y_train_orig, X_val, y_val,
    params=xgb_params,
    tags={'iteration': '8', 'model_family': 'XGBoost'}
)

# COMMAND ----------

# Model 9: XGBoost - SMOTE
xgb_smote = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

xgb_smote_params = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'gamma': 0.1
}

model_xgb_smote, metrics_xgb_smote = train_and_log_model(
    xgb_smote,
    "XGBoost_SMOTE",
    X_train_smote, y_train_smote, X_val, y_val,
    params=xgb_smote_params,
    tags={'iteration': '9', 'model_family': 'XGBoost'},
    use_smote=True
)

# COMMAND ----------

# Model 10: XGBoost - Optimized
xgb_opt = XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.03,
    scale_pos_weight=(y_train_orig == 0).sum() / (y_train_orig == 1).sum(),
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0.2,
    min_child_weight=3,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

xgb_opt_params = {
    'n_estimators': 300,
    'max_depth': 10,
    'learning_rate': 0.03,
    'scale_pos_weight': float((y_train_orig == 0).sum() / (y_train_orig == 1).sum()),
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'gamma': 0.2,
    'min_child_weight': 3
}

model_xgb_opt, metrics_xgb_opt = train_and_log_model(
    xgb_opt,
    "XGBoost_Optimized",
    X_train_orig, y_train_orig, X_val, y_val,
    params=xgb_opt_params,
    tags={'iteration': '10', 'model_family': 'XGBoost'}
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. LightGBM Models

# COMMAND ----------

# Model 11: LightGBM - Balanced
lgbm_model = LGBMClassifier(
    n_estimators=100,
    max_depth=7,
    learning_rate=0.1,
    num_leaves=31,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)

lgbm_params = {
    'n_estimators': 100,
    'max_depth': 7,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'class_weight': 'balanced'
}

model_lgbm, metrics_lgbm = train_and_log_model(
    lgbm_model,
    "LightGBM_Balanced",
    X_train_orig, y_train_orig, X_val, y_val,
    params=lgbm_params,
    tags={'iteration': '11', 'model_family': 'LightGBM'}
)

# COMMAND ----------

# Model 12: LightGBM - SMOTE
lgbm_smote = LGBMClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.05,
    num_leaves=63,
    random_state=42,
    verbose=-1
)

lgbm_smote_params = {
    'n_estimators': 200,
    'max_depth': 10,
    'learning_rate': 0.05,
    'num_leaves': 63
}

model_lgbm_smote, metrics_lgbm_smote = train_and_log_model(
    lgbm_smote,
    "LightGBM_SMOTE",
    X_train_smote, y_train_smote, X_val, y_val,
    params=lgbm_smote_params,
    tags={'iteration': '12', 'model_family': 'LightGBM'},
    use_smote=True
)

# COMMAND ----------

# Model 13: LightGBM - Optimized
lgbm_opt = LGBMClassifier(
    n_estimators=300,
    max_depth=12,
    learning_rate=0.03,
    num_leaves=127,
    class_weight='balanced',
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

lgbm_opt_params = {
    'n_estimators': 300,
    'max_depth': 12,
    'learning_rate': 0.03,
    'num_leaves': 127,
    'class_weight': 'balanced',
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

model_lgbm_opt, metrics_lgbm_opt = train_and_log_model(
    lgbm_opt,
    "LightGBM_Optimized",
    X_train_orig, y_train_orig, X_val, y_val,
    params=lgbm_opt_params,
    tags={'iteration': '13', 'model_family': 'LightGBM'}
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. CatBoost Models

# COMMAND ----------

# Model 14: CatBoost - Balanced
catboost_model = CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    auto_class_weights='Balanced',
    random_seed=42,
    verbose=False
)

catboost_params = {
    'iterations': 100,
    'depth': 6,
    'learning_rate': 0.1,
    'auto_class_weights': 'Balanced'
}

model_catboost, metrics_catboost = train_and_log_model(
    catboost_model,
    "CatBoost_Balanced",
    X_train_orig.values, y_train_orig.values, X_val.values, y_val.values,
    params=catboost_params,
    tags={'iteration': '14', 'model_family': 'CatBoost'}
)

# COMMAND ----------

# Model 15: CatBoost - SMOTE
catboost_smote = CatBoostClassifier(
    iterations=200,
    depth=8,
    learning_rate=0.05,
    random_seed=42,
    verbose=False
)

catboost_smote_params = {
    'iterations': 200,
    'depth': 8,
    'learning_rate': 0.05
}

model_catboost_smote, metrics_catboost_smote = train_and_log_model(
    catboost_smote,
    "CatBoost_SMOTE",
    X_train_smote.values, y_train_smote.values, X_val.values, y_val.values,
    params=catboost_smote_params,
    tags={'iteration': '15', 'model_family': 'CatBoost'},
    use_smote=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Additional Algorithms

# COMMAND ----------

# Model 16: K-Nearest Neighbors (SMOTE only - too slow on full data)
knn_model = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    n_jobs=-1
)

knn_params = {
    'n_neighbors': 5,
    'weights': 'distance',
    'metric': 'minkowski'
}

# Use smaller subset for KNN
X_train_knn = X_train_smote[:50000]
y_train_knn = y_train_smote[:50000]

model_knn, metrics_knn = train_and_log_model(
    knn_model,
    "KNN_SMOTE",
    X_train_knn, y_train_knn, X_val, y_val,
    params=knn_params,
    tags={'iteration': '16', 'note': 'subset_training'},
    use_smote=True
)

# COMMAND ----------

# Model 17: Naive Bayes
nb_model = GaussianNB()

nb_params = {
    'var_smoothing': 1e-9
}

model_nb, metrics_nb = train_and_log_model(
    nb_model,
    "Naive_Bayes_SMOTE",
    X_train_smote, y_train_smote, X_val, y_val,
    params=nb_params,
    tags={'iteration': '17'},
    use_smote=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Hyperparameter Tuned Models

# COMMAND ----------

# Model 18: XGBoost - Fine-tuned
xgb_tuned = XGBClassifier(
    n_estimators=350,
    max_depth=9,
    learning_rate=0.025,
    scale_pos_weight=(y_train_orig == 0).sum() / (y_train_orig == 1).sum(),
    subsample=0.88,
    colsample_bytree=0.88,
    gamma=0.15,
    min_child_weight=2,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

xgb_tuned_params = {
    'n_estimators': 350,
    'max_depth': 9,
    'learning_rate': 0.025,
    'scale_pos_weight': float((y_train_orig == 0).sum() / (y_train_orig == 1).sum()),
    'subsample': 0.88,
    'colsample_bytree': 0.88,
    'gamma': 0.15,
    'min_child_weight': 2,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}

model_xgb_tuned, metrics_xgb_tuned = train_and_log_model(
    xgb_tuned,
    "XGBoost_Tuned",
    X_train_orig, y_train_orig, X_val, y_val,
    params=xgb_tuned_params,
    tags={'iteration': '18', 'model_family': 'XGBoost', 'hyperparameter_tuned': 'true'}
)

# COMMAND ----------

# Model 19: LightGBM - Fine-tuned
lgbm_tuned = LGBMClassifier(
    n_estimators=350,
    max_depth=11,
    learning_rate=0.025,
    num_leaves=100,
    class_weight='balanced',
    min_child_samples=15,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1
)

lgbm_tuned_params = {
    'n_estimators': 350,
    'max_depth': 11,
    'learning_rate': 0.025,
    'num_leaves': 100,
    'class_weight': 'balanced',
    'min_child_samples': 15,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}

model_lgbm_tuned, metrics_lgbm_tuned = train_and_log_model(
    lgbm_tuned,
    "LightGBM_Tuned",
    X_train_orig, y_train_orig, X_val, y_val,
    params=lgbm_tuned_params,
    tags={'iteration': '19', 'model_family': 'LightGBM', 'hyperparameter_tuned': 'true'}
)

# COMMAND ----------

# Model 20: Random Forest - Fine-tuned
rf_tuned = RandomForestClassifier(
    n_estimators=400,
    max_depth=30,
    min_samples_split=8,
    min_samples_leaf=4,
    max_features='log2',
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

rf_tuned_params = {
    'n_estimators': 400,
    'max_depth': 30,
    'min_samples_split': 8,
    'min_samples_leaf': 4,
    'max_features': 'log2',
    'class_weight': 'balanced_subsample'
}

model_rf_tuned, metrics_rf_tuned = train_and_log_model(
    rf_tuned,
    "Random_Forest_Tuned",
    X_train_orig, y_train_orig, X_val, y_val,
    params=rf_tuned_params,
    tags={'iteration': '20', 'model_family': 'RandomForest', 'hyperparameter_tuned': 'true'}
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Compare All Models

# COMMAND ----------

# Get all runs from the experiment
experiment = mlflow.get_experiment_by_name(experiment_name)
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Sort by F1 score
runs_sorted = runs.sort_values('metrics.val_f1', ascending=False)

# Display top 10 models
print("=" * 100)
print("TOP 10 MODELS BY F1-SCORE")
print("=" * 100)

display(runs_sorted[['tags.model_type', 'metrics.val_accuracy', 'metrics.val_precision', 
                     'metrics.val_recall', 'metrics.val_f1', 'metrics.val_roc_auc']].head(10))

# COMMAND ----------

# Visualize model comparison
top_models = runs_sorted.head(20)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# F1 Score
axes[0, 0].barh(range(len(top_models)), top_models['metrics.val_f1'])
axes[0, 0].set_yticks(range(len(top_models)))
axes[0, 0].set_yticklabels(top_models['tags.model_type'], fontsize=9)
axes[0, 0].set_xlabel('F1 Score')
axes[0, 0].set_title('Model Comparison - F1 Score', fontweight='bold')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(axis='x', alpha=0.3)

# ROC-AUC
axes[0, 1].barh(range(len(top_models)), top_models['metrics.val_roc_auc'], color='orange')
axes[0, 1].set_yticks(range(len(top_models)))
axes[0, 1].set_yticklabels(top_models['tags.model_type'], fontsize=9)
axes[0, 1].set_xlabel('ROC-AUC')
axes[0, 1].set_title('Model Comparison - ROC-AUC', fontweight='bold')
axes[0, 1].invert_yaxis()
axes[0, 1].grid(axis='x', alpha=0.3)

# Precision vs Recall
axes[1, 0].scatter(top_models['metrics.val_recall'], top_models['metrics.val_precision'], 
                   s=100, alpha=0.6)
for idx, model in enumerate(top_models['tags.model_type'].head(10)):
    axes[1, 0].annotate(model, 
                       (top_models.iloc[idx]['metrics.val_recall'], 
                        top_models.iloc[idx]['metrics.val_precision']),
                       fontsize=7, alpha=0.7)
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Precision vs Recall Trade-off', fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Training Time
axes[1, 1].barh(range(len(top_models)), top_models['metrics.training_time_seconds'], color='green')
axes[1, 1].set_yticks(range(len(top_models)))
axes[1, 1].set_yticklabels(top_models['tags.model_type'], fontsize=9)
axes[1, 1].set_xlabel('Training Time (seconds)')
axes[1, 1].set_title('Model Training Time', fontweight='bold')
axes[1, 1].invert_yaxis()
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/dbfs/FileStore/fraud_detection/visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Select Champion Model

# COMMAND ----------

# Get best model based on F1 score
best_run = runs_sorted.iloc[0]
champion_model_name = best_run['tags.model_type']
champion_run_id = best_run['run_id']

print("=" * 100)
print("🏆 CHAMPION MODEL SELECTED")
print("=" * 100)
print(f"\nModel: {champion_model_name}")
print(f"Run ID: {champion_run_id}")
print(f"\nPerformance Metrics:")
print(f"  Accuracy:  {best_run['metrics.val_accuracy']:.4f}")
print(f"  Precision: {best_run['metrics.val_precision']:.4f}")
print(f"  Recall:    {best_run['metrics.val_recall']:.4f}")
print(f"  F1-Score:  {best_run['metrics.val_f1']:.4f}")
print(f"  ROC-AUC:   {best_run['metrics.val_roc_auc']:.4f}")
print("\n" + "=" * 100)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Final Evaluation on Test Set

# COMMAND ----------

# Load champion model
champion_model_uri = f"runs:/{champion_run_id}/model"
champion_model = mlflow.sklearn.load_model(champion_model_uri)

# Evaluate on test set
test_metrics, test_pred, test_proba = evaluate_model(champion_model, X_test, y_test, "test")

print("=" * 100)
print("FINAL TEST SET EVALUATION - CHAMPION MODEL")
print("=" * 100)
print(f"\nTest Set Metrics:")
print(f"  Accuracy:  {test_metrics['test_accuracy']:.4f}")
print(f"  Precision: {test_metrics['test_precision']:.4f}")
print(f"  Recall:    {test_metrics['test_recall']:.4f}")
print(f"  F1-Score:  {test_metrics['test_f1']:.4f}")
print(f"  ROC-AUC:   {test_metrics['test_roc_auc']:.4f}")
print("\n" + "=" * 100)

# Log test metrics to the champion run
with mlflow.start_run(run_id=champion_run_id):
    mlflow.log_metrics(test_metrics)

# COMMAND ----------

# Create test set visualizations
test_cm = confusion_matrix(y_test, test_pred)
test_cm_fig = plot_confusion_matrix(test_cm, f"{champion_model_name} - Test Set Confusion Matrix")
plt.savefig('/dbfs/FileStore/fraud_detection/visualizations/champion_test_confusion_matrix.png', 
            dpi=300, bbox_inches='tight')
plt.show()

test_roc_fig = plot_roc_curve(y_test, test_proba, f"{champion_model_name} - Test Set ROC Curve")
plt.savefig('/dbfs/FileStore/fraud_detection/visualizations/champion_test_roc_curve.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Summary

# COMMAND ----------

print("=" * 100)
print("MODEL TRAINING SUMMARY")
print("=" * 100)
print(f"\n📊 Experiments Tracked: {len(runs)}")
print(f"🏆 Champion Model: {champion_model_name}")
print(f"📈 Best F1-Score: {best_run['metrics.val_f1']:.4f}")
print(f"🎯 Test F1-Score: {test_metrics['test_f1']:.4f}")
print(f"\n✅ All models logged to MLflow")
print(f"📁 Experiment: {experiment_name}")
print("\n" + "=" * 100)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC ✅ **20+ models trained and tracked**
# MAGIC ✅ **Champion model selected**
# MAGIC ✅ **All experiments logged to MLflow**
# MAGIC 
# MAGIC Next: `04_model_registry.py` - Register models and prepare for deployment!

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **Notebook Complete!** ✓
# MAGIC 
# MAGIC Proceed to: `04_model_registry.py`
