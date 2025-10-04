# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Feature Engineering & Preprocessing
# MAGIC 
# MAGIC ## Objectives
# MAGIC - Load processed data from previous notebook
# MAGIC - Create engineered features
# MAGIC - Handle class imbalance
# MAGIC - Scale and normalize features
# MAGIC - Create train/validation/test splits
# MAGIC - Save preprocessed data for model training

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Imports

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime

print("Libraries imported successfully!")
print(f"Execution Time: {datetime.now()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Processed Data

# COMMAND ----------

# Load data from previous notebook
df = pd.read_parquet('/dbfs/FileStore/fraud_detection/processed/creditcard_processed.parquet')

print(f"Data loaded: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()}")
print(f"Normal cases: {len(df) - df['Class'].sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature Engineering

# COMMAND ----------

# Create copy for feature engineering
df_engineered = df.copy()

# Feature 1: Hour of day (already created in EDA notebook)
if 'Hour' not in df_engineered.columns:
    df_engineered['Hour'] = (df_engineered['Time'] / 3600) % 24

# Feature 2: Time period (Night, Morning, Afternoon, Evening)
def get_time_period(hour):
    if 0 <= hour < 6:
        return 0  # Night
    elif 6 <= hour < 12:
        return 1  # Morning
    elif 12 <= hour < 18:
        return 2  # Afternoon
    else:
        return 3  # Evening

df_engineered['Time_Period'] = df_engineered['Hour'].apply(get_time_period)

# Feature 3: Amount bins (Low, Medium, High, Very High)
df_engineered['Amount_Bin'] = pd.cut(df_engineered['Amount'], 
                                      bins=[0, 50, 200, 1000, np.inf],
                                      labels=[0, 1, 2, 3])
df_engineered['Amount_Bin'] = df_engineered['Amount_Bin'].astype(int)

# Feature 4: Log transformation of Amount (handle zero values)
df_engineered['Amount_Log'] = np.log1p(df_engineered['Amount'])

# Feature 5: Amount per hour
df_engineered['Amount_Per_Hour'] = df_engineered['Amount'] / (df_engineered['Hour'] + 1)

# Feature 6: Is weekend (assuming time starts on a Monday)
df_engineered['Day_Of_Week'] = ((df_engineered['Time'] // 86400) % 7).astype(int)
df_engineered['Is_Weekend'] = (df_engineered['Day_Of_Week'] >= 5).astype(int)

print("Engineered features created:")
print("- Hour")
print("- Time_Period")
print("- Amount_Bin")
print("- Amount_Log")
print("- Amount_Per_Hour")
print("- Day_Of_Week")
print("- Is_Weekend")

# COMMAND ----------

# Feature importance: V feature interactions
# Create a few high-value interactions based on correlation analysis
v_features = [col for col in df.columns if col.startswith('V')]

# Top correlated features from EDA
df_engineered['V17_V14'] = df_engineered['V17'] * df_engineered['V14']
df_engineered['V12_V10'] = df_engineered['V12'] * df_engineered['V10']
df_engineered['V17_V12'] = df_engineered['V17'] * df_engineered['V12']

# Polynomial features for top 3 most correlated
df_engineered['V17_squared'] = df_engineered['V17'] ** 2
df_engineered['V14_squared'] = df_engineered['V14'] ** 2
df_engineered['V12_squared'] = df_engineered['V12'] ** 2

print("\nInteraction features created:")
print("- V17_V14, V12_V10, V17_V12")
print("- V17_squared, V14_squared, V12_squared")

# COMMAND ----------

# Display engineered dataset
print(f"Original features: {len(df.columns)}")
print(f"Engineered features: {len(df_engineered.columns)}")
print(f"New features added: {len(df_engineered.columns) - len(df.columns)}")

display(df_engineered.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train/Validation/Test Split

# COMMAND ----------

# Separate features and target
X = df_engineered.drop(['Class', 'Time'], axis=1)  # Drop Time as we have derived features
y = df_engineered['Class']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# COMMAND ----------

# Split: 60% train, 20% validation, 20% test
# First split: 80% train+val, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: 75% of temp = 60% total train, 25% of temp = 20% total val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print("Dataset Split:")
print(f"Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  - Normal: {(y_train == 0).sum():,}")
print(f"  - Fraud: {(y_train == 1).sum():,}")
print(f"\nValidation: {X_val.shape[0]:,} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"  - Normal: {(y_val == 0).sum():,}")
print(f"  - Fraud: {(y_val == 1).sum():,}")
print(f"\nTest: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"  - Normal: {(y_test == 0).sum():,}")
print(f"  - Fraud: {(y_test == 1).sum():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Feature Scaling

# COMMAND ----------

# Initialize scaler (RobustScaler is better for outliers)
scaler = RobustScaler()

# Fit on training data only
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("Feature scaling completed using RobustScaler")
print(f"Scaled train shape: {X_train_scaled.shape}")

# COMMAND ----------

# Verify scaling
print("Before scaling (sample):")
print(X_train[['Amount', 'Amount_Log', 'V1', 'V2']].head())

print("\nAfter scaling (sample):")
print(X_train_scaled[['Amount', 'Amount_Log', 'V1', 'V2']].head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Handle Class Imbalance

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 1: Save Original Imbalanced Data (for baseline)

# COMMAND ----------

# Save original splits (imbalanced)
output_path = '/dbfs/FileStore/fraud_detection/prepared/'
os.makedirs(output_path, exist_ok=True)

# Save as pickle for exact numpy array preservation
import pickle

with open(f'{output_path}original_train.pkl', 'wb') as f:
    pickle.dump((X_train_scaled, y_train), f)

with open(f'{output_path}original_val.pkl', 'wb') as f:
    pickle.dump((X_val_scaled, y_val), f)

with open(f'{output_path}original_test.pkl', 'wb') as f:
    pickle.dump((X_test_scaled, y_test), f)

print("Original (imbalanced) data saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 2: SMOTE (Synthetic Minority Over-sampling)

# COMMAND ----------

# Apply SMOTE to training data
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # 50% fraud after resampling
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print("SMOTE Resampling Results:")
print(f"Original training size: {len(X_train_scaled):,}")
print(f"  - Normal: {(y_train == 0).sum():,}")
print(f"  - Fraud: {(y_train == 1).sum():,}")
print(f"\nAfter SMOTE: {len(X_train_smote):,}")
print(f"  - Normal: {(y_train_smote == 0).sum():,}")
print(f"  - Fraud: {(y_train_smote == 1).sum():,}")
print(f"  - New ratio: 1:{(y_train_smote == 0).sum() / (y_train_smote == 1).sum():.1f}")

# COMMAND ----------

# Save SMOTE data
with open(f'{output_path}smote_train.pkl', 'wb') as f:
    pickle.dump((X_train_smote, y_train_smote), f)

print("SMOTE data saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 3: ADASYN (Adaptive Synthetic Sampling)

# COMMAND ----------

# Apply ADASYN (focuses on hard-to-learn samples)
adasyn = ADASYN(sampling_strategy=0.5, random_state=42)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train_scaled, y_train)

print("ADASYN Resampling Results:")
print(f"After ADASYN: {len(X_train_adasyn):,}")
print(f"  - Normal: {(y_train_adasyn == 0).sum():,}")
print(f"  - Fraud: {(y_train_adasyn == 1).sum():,}")

# Save ADASYN data
with open(f'{output_path}adasyn_train.pkl', 'wb') as f:
    pickle.dump((X_train_adasyn, y_train_adasyn), f)

print("ADASYN data saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 4: Random Undersampling

# COMMAND ----------

# Apply random undersampling
rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train_scaled, y_train)

print("Random Undersampling Results:")
print(f"After undersampling: {len(X_train_rus):,}")
print(f"  - Normal: {(y_train_rus == 0).sum():,}")
print(f"  - Fraud: {(y_train_rus == 1).sum():,}")

# Save undersampled data
with open(f'{output_path}undersampled_train.pkl', 'wb') as f:
    pickle.dump((X_train_rus, y_train_rus), f)

print("Undersampled data saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 5: SMOTE + Tomek Links (Hybrid)

# COMMAND ----------

# Apply SMOTE + Tomek (combines oversampling and cleaning)
smt = SMOTETomek(sampling_strategy=0.5, random_state=42)
X_train_smotetomek, y_train_smotetomek = smt.fit_resample(X_train_scaled, y_train)

print("SMOTE + Tomek Results:")
print(f"After SMOTE + Tomek: {len(X_train_smotetomek):,}")
print(f"  - Normal: {(y_train_smotetomek == 0).sum():,}")
print(f"  - Fraud: {(y_train_smotetomek == 1).sum():,}")

# Save SMOTE + Tomek data
with open(f'{output_path}smotetomek_train.pkl', 'wb') as f:
    pickle.dump((X_train_smotetomek, y_train_smotetomek), f)

print("SMOTE + Tomek data saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Visualize Resampling Results

# COMMAND ----------

# Compare class distributions
strategies = ['Original', 'SMOTE', 'ADASYN', 'Undersampling', 'SMOTE+Tomek']
fraud_counts = [
    (y_train == 1).sum(),
    (y_train_smote == 1).sum(),
    (y_train_adasyn == 1).sum(),
    (y_train_rus == 1).sum(),
    (y_train_smotetomek == 1).sum()
]
normal_counts = [
    (y_train == 0).sum(),
    (y_train_smote == 0).sum(),
    (y_train_adasyn == 0).sum(),
    (y_train_rus == 0).sum(),
    (y_train_smotetomek == 0).sum()
]

# Create comparison plot
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(strategies))
width = 0.35

bars1 = ax.bar(x - width/2, normal_counts, width, label='Normal', color='#2ecc71', alpha=0.8)
bars2 = ax.bar(x + width/2, fraud_counts, width, label='Fraud', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Resampling Strategy', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Class Distribution Across Resampling Strategies', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(strategies, rotation=15, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('/dbfs/FileStore/fraud_detection/visualizations/resampling_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save Scaler and Feature Names

# COMMAND ----------

# Save scaler for inference
with open(f'{output_path}scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Scaler saved")

# Save feature names
feature_names = X_train.columns.tolist()
with open(f'{output_path}feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print(f"Feature names saved: {len(feature_names)} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Create Class Weight Dictionary

# COMMAND ----------

# For models that support class weights (instead of resampling)
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(zip(np.unique(y_train), class_weights))

print("Class Weights (for balanced loss):")
print(f"  Class 0 (Normal): {class_weight_dict[0]:.4f}")
print(f"  Class 1 (Fraud): {class_weight_dict[1]:.4f}")

# Save class weights
with open(f'{output_path}class_weights.pkl', 'wb') as f:
    pickle.dump(class_weight_dict, f)

print("Class weights saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Save Data Summary

# COMMAND ----------

# Create metadata file
metadata = {
    'total_samples': len(df_engineered),
    'total_features': len(X.columns),
    'feature_names': feature_names,
    'train_size': len(X_train),
    'val_size': len(X_val),
    'test_size': len(X_test),
    'train_fraud_count': int((y_train == 1).sum()),
    'val_fraud_count': int((y_val == 1).sum()),
    'test_fraud_count': int((y_test == 1).sum()),
    'scaling_method': 'RobustScaler',
    'resampling_strategies': strategies,
    'class_weights': {int(k): float(v) for k, v in class_weight_dict.items()},
    'timestamp': datetime.now().isoformat()
}

import json
with open(f'{output_path}metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Metadata saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Summary

# COMMAND ----------

print("=" * 80)
print("FEATURE ENGINEERING & PREPROCESSING SUMMARY")
print("=" * 80)
print(f"\n📊 Dataset Statistics:")
print(f"   Total Samples: {len(df_engineered):,}")
print(f"   Total Features: {len(X.columns)}")
print(f"   Engineered Features: {len(X.columns) - len(df.columns) + 2}")  # +2 for Time and Class

print(f"\n📂 Data Splits:")
print(f"   Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Validation: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
print(f"   Test: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

print(f"\n⚖️ Resampling Strategies:")
for i, strategy in enumerate(strategies):
    print(f"   {i+1}. {strategy}: {normal_counts[i]:,} normal, {fraud_counts[i]:,} fraud")

print(f"\n📁 Saved Files:")
print(f"   Location: {output_path}")
print(f"   - Original data (imbalanced)")
print(f"   - SMOTE resampled data")
print(f"   - ADASYN resampled data")
print(f"   - Undersampled data")
print(f"   - SMOTE+Tomek data")
print(f"   - Scaler (RobustScaler)")
print(f"   - Feature names")
print(f"   - Class weights")
print(f"   - Metadata")

print("\n" + "=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Next Steps
# MAGIC 
# MAGIC Data is now ready for model training! We have:
# MAGIC 
# MAGIC ✅ **Engineered features** with domain knowledge
# MAGIC ✅ **Multiple resampling strategies** to compare
# MAGIC ✅ **Proper train/val/test splits** with stratification
# MAGIC ✅ **Scaled features** using RobustScaler
# MAGIC ✅ **Saved artifacts** for reproducibility
# MAGIC 
# MAGIC Next: `03_model_training.py` - Train models with MLflow tracking!

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **Notebook Complete!** ✓
# MAGIC 
# MAGIC Proceed to: `03_model_training.py`
