# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Data Ingestion & Exploratory Data Analysis
# MAGIC 
# MAGIC ## Objectives
# MAGIC - Load credit card fraud dataset from Kaggle
# MAGIC - Perform exploratory data analysis
# MAGIC - Understand class imbalance
# MAGIC - Visualize feature distributions
# MAGIC - Save processed data to DBFS

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Databricks specific
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Libraries imported successfully!")
print(f"Execution Time: {datetime.now()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Dataset Download from Kaggle
# MAGIC 
# MAGIC **Note**: You'll need to:
# MAGIC 1. Get your Kaggle API credentials from https://www.kaggle.com/settings
# MAGIC 2. Upload kaggle.json to DBFS or set as secrets
# MAGIC 3. Install kaggle library

# COMMAND ----------

# MAGIC %pip install kaggle

# COMMAND ----------

# Option 1: Using Kaggle API (recommended)
import os
import zipfile

# Set Kaggle credentials (use Databricks secrets in production)
os.environ['KAGGLE_USERNAME'] = 'your_kaggle_username'  # Replace or use dbutils.secrets.get()
os.environ['KAGGLE_KEY'] = 'your_kaggle_key'  # Replace or use dbutils.secrets.get()

# Download dataset
dataset_path = '/dbfs/FileStore/fraud_detection/raw/'
os.makedirs(dataset_path, exist_ok=True)

# Download from Kaggle
import kaggle
kaggle.api.dataset_download_files(
    'mlg-ulb/creditcardfraud',
    path=dataset_path,
    unzip=True
)

print(f"Dataset downloaded to: {dataset_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alternative: Manual Upload
# MAGIC 
# MAGIC If you prefer to manually upload the dataset:
# MAGIC 1. Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# MAGIC 2. Upload to DBFS: Data > Add Data > Browse
# MAGIC 3. Update the path below

# COMMAND ----------

# Load dataset
df = pd.read_csv('/dbfs/FileStore/fraud_detection/raw/creditcard.csv')

print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Initial Data Inspection

# COMMAND ----------

# Display first few rows
display(df.head(10))

# COMMAND ----------

# Dataset info
print("=" * 80)
print("DATASET INFORMATION")
print("=" * 80)
print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print("\n" + "=" * 80)

df.info()

# COMMAND ----------

# Statistical summary
display(df.describe())

# COMMAND ----------

# Check for missing values
missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

print("Missing Values Summary:")
if len(missing_data) == 0:
    print("✓ No missing values found!")
else:
    display(missing_data)

# COMMAND ----------

# Check for duplicates
duplicate_count = df.duplicated().sum()
print(f"Duplicate Rows: {duplicate_count:,}")

if duplicate_count > 0:
    print(f"Percentage: {(duplicate_count / len(df)) * 100:.2f}%")
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape[0]:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Class Imbalance Analysis

# COMMAND ----------

# Class distribution
class_distribution = df['Class'].value_counts()
fraud_percentage = (class_distribution[1] / len(df)) * 100

print("=" * 80)
print("CLASS DISTRIBUTION")
print("=" * 80)
print(f"\nNormal Transactions: {class_distribution[0]:,} ({100 - fraud_percentage:.3f}%)")
print(f"Fraudulent Transactions: {class_distribution[1]:,} ({fraud_percentage:.3f}%)")
print(f"\nImbalance Ratio: 1:{int(class_distribution[0] / class_distribution[1])}")
print("=" * 80)

# COMMAND ----------

# Visualize class distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Count plot
class_distribution.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Class Distribution (Absolute)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Class (0=Normal, 1=Fraud)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_xticklabels(['Normal', 'Fraud'], rotation=0)
axes[0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(class_distribution):
    axes[0].text(i, v + 5000, f'{v:,}', ha='center', va='bottom', fontweight='bold')

# Pie chart
colors = ['#2ecc71', '#e74c3c']
explode = (0, 0.1)
axes[1].pie(class_distribution, labels=['Normal', 'Fraud'], autopct='%1.3f%%',
            colors=colors, explode=explode, shadow=True, startangle=90)
axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('/dbfs/FileStore/fraud_detection/visualizations/class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Feature Analysis

# COMMAND ----------

# Time feature analysis
print("Time Feature Statistics:")
print(df['Time'].describe())

# Convert Time to hours
df['Hour'] = (df['Time'] / 3600) % 24

# COMMAND ----------

# Transaction distribution over time
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Normal transactions over time
normal_time = df[df['Class'] == 0]['Hour']
axes[0].hist(normal_time, bins=24, color='#2ecc71', alpha=0.7, edgecolor='black')
axes[0].set_title('Normal Transactions - Distribution Over Time', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Hour of Day', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

# Fraud transactions over time
fraud_time = df[df['Class'] == 1]['Hour']
axes[1].hist(fraud_time, bins=24, color='#e74c3c', alpha=0.7, edgecolor='black')
axes[1].set_title('Fraudulent Transactions - Distribution Over Time', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Hour of Day', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/dbfs/FileStore/fraud_detection/visualizations/time_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

# Amount feature analysis
print("Amount Feature Statistics:")
print(df['Amount'].describe())

print("\nAmount Statistics by Class:")
print(df.groupby('Class')['Amount'].describe())

# COMMAND ----------

# Amount distribution visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Box plot
df.boxplot(column='Amount', by='Class', ax=axes[0])
axes[0].set_title('Amount Distribution by Class', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Class (0=Normal, 1=Fraud)', fontsize=12)
axes[0].set_ylabel('Amount', fontsize=12)
plt.suptitle('')  # Remove auto-generated title

# Histogram (log scale)
axes[1].hist(df[df['Class'] == 0]['Amount'], bins=50, alpha=0.5, label='Normal', color='#2ecc71')
axes[1].hist(df[df['Class'] == 1]['Amount'], bins=50, alpha=0.5, label='Fraud', color='#e74c3c')
axes[1].set_xlabel('Amount', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Amount Distribution (Both Classes)', fontsize=14, fontweight='bold')
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/dbfs/FileStore/fraud_detection/visualizations/amount_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. PCA Features (V1-V28) Analysis

# COMMAND ----------

# Analyze V features
v_features = [col for col in df.columns if col.startswith('V')]
print(f"PCA Features: {len(v_features)}")

# Statistical summary for V features
v_stats = df[v_features].describe()
display(v_stats)

# COMMAND ----------

# Correlation of V features with Class
correlations = df[v_features + ['Class']].corr()['Class'].drop('Class').sort_values(ascending=False)

print("Top 10 Most Correlated Features with Fraud:")
print(correlations.head(10))

print("\nTop 10 Most Negatively Correlated Features with Fraud:")
print(correlations.tail(10))

# COMMAND ----------

# Visualize feature correlations
fig, ax = plt.subplots(figsize=(12, 8))

correlations_sorted = correlations.sort_values()
colors = ['#e74c3c' if x > 0 else '#3498db' for x in correlations_sorted]

correlations_sorted.plot(kind='barh', ax=ax, color=colors)
ax.set_title('Feature Correlation with Fraud Class', fontsize=16, fontweight='bold')
ax.set_xlabel('Correlation Coefficient', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/dbfs/FileStore/fraud_detection/visualizations/feature_correlations.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

# Feature distributions for top correlated features
top_features = correlations.abs().nlargest(9).index.tolist()

fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, feature in enumerate(top_features):
    axes[idx].hist(df[df['Class'] == 0][feature], bins=50, alpha=0.5, label='Normal', color='#2ecc71', density=True)
    axes[idx].hist(df[df['Class'] == 1][feature], bins=50, alpha=0.5, label='Fraud', color='#e74c3c', density=True)
    axes[idx].set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(feature, fontsize=10)
    axes[idx].set_ylabel('Density', fontsize=10)
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/dbfs/FileStore/fraud_detection/visualizations/top_features_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Correlation Matrix

# COMMAND ----------

# Sample for correlation matrix (too large otherwise)
sample_size = 10000
df_sample = df.sample(n=sample_size, random_state=42)

# Compute correlation matrix
correlation_matrix = df_sample[v_features].corr()

# Plot correlation heatmap
fig, ax = plt.subplots(figsize=(20, 16))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            ax=ax, vmin=-1, vmax=1)
ax.set_title('Feature Correlation Matrix (PCA Features)', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/dbfs/FileStore/fraud_detection/visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Data Quality Checks

# COMMAND ----------

# Check for outliers in Amount
Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Amount'] < Q1 - 1.5 * IQR) | (df['Amount'] > Q3 + 1.5 * IQR)]

print(f"Outliers in Amount: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")

# COMMAND ----------

# Check value ranges for V features
print("Value Ranges for V Features:")
for feature in v_features[:5]:  # Show first 5 as example
    min_val = df[feature].min()
    max_val = df[feature].max()
    print(f"{feature}: [{min_val:.4f}, {max_val:.4f}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Save Processed Data

# COMMAND ----------

# Save to DBFS for next notebook
output_path = '/dbfs/FileStore/fraud_detection/processed/'
os.makedirs(output_path, exist_ok=True)

# Save as CSV
df.to_csv(f'{output_path}creditcard_processed.csv', index=False)
print(f"Data saved to: {output_path}creditcard_processed.csv")

# Also save as Parquet for better performance
df.to_parquet(f'{output_path}creditcard_processed.parquet', index=False)
print(f"Data saved to: {output_path}creditcard_processed.parquet")

# COMMAND ----------

# Save as Delta table (Databricks optimized format)
spark_df = spark.createDataFrame(df)
spark_df.write.format("delta").mode("overwrite").save("/FileStore/fraud_detection/delta/creditcard")
print("Data saved as Delta table")

# Register as table
spark_df.write.format("delta").mode("overwrite").saveAsTable("fraud_detection.creditcard")
print("Data registered as table: fraud_detection.creditcard")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Summary Statistics

# COMMAND ----------

# Create comprehensive summary
summary = {
    'Total Transactions': len(df),
    'Normal Transactions': class_distribution[0],
    'Fraudulent Transactions': class_distribution[1],
    'Fraud Percentage': f"{fraud_percentage:.3f}%",
    'Imbalance Ratio': f"1:{int(class_distribution[0] / class_distribution[1])}",
    'Features': len(df.columns) - 1,  # Excluding Class
    'PCA Features': len(v_features),
    'Missing Values': df.isnull().sum().sum(),
    'Duplicates Removed': duplicate_count,
    'Time Range (hours)': f"{df['Time'].min() / 3600:.1f} - {df['Time'].max() / 3600:.1f}",
    'Amount Range': f"${df['Amount'].min():.2f} - ${df['Amount'].max():.2f}",
    'Mean Amount (Normal)': f"${df[df['Class']==0]['Amount'].mean():.2f}",
    'Mean Amount (Fraud)': f"${df[df['Class']==1]['Amount'].mean():.2f}"
}

print("=" * 80)
print("DATASET SUMMARY")
print("=" * 80)
for key, value in summary.items():
    print(f"{key:.<40} {value}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Key Findings
# MAGIC 
# MAGIC ### Dataset Characteristics:
# MAGIC - **Highly Imbalanced**: Only 0.172% fraud cases (492 out of 284,807)
# MAGIC - **PCA Transformed**: Features V1-V28 are already anonymized via PCA
# MAGIC - **Clean Data**: No missing values
# MAGIC - **Time Span**: Transactions over 48 hours
# MAGIC 
# MAGIC ### Observations:
# MAGIC 1. **Class Imbalance**: Requires special handling (SMOTE, class weights, undersampling)
# MAGIC 2. **Feature Correlations**: Several V features show strong correlation with fraud
# MAGIC 3. **Amount Distribution**: Fraud transactions have different amount patterns
# MAGIC 4. **Time Patterns**: Some temporal patterns visible in fraud vs normal transactions
# MAGIC 
# MAGIC ### Next Steps:
# MAGIC - Feature engineering and scaling
# MAGIC - Handle class imbalance
# MAGIC - Train/validation/test split
# MAGIC - Model training with MLflow tracking

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **Notebook Complete!** ✓
# MAGIC 
# MAGIC Proceed to: `02_feature_engineering.py`
