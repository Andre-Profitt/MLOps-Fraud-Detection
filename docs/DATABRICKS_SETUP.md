# Databricks Setup Guide

Complete guide to setting up your Databricks workspace for the MLflow FinTech Fraud Detection Pipeline.

## Table of Contents
- [Option 1: Databricks Community Edition (Free)](#option-1-databricks-community-edition-free)
- [Option 2: AWS Databricks](#option-2-aws-databricks)
- [Option 3: Azure Databricks](#option-3-azure-databricks)
- [Option 4: GCP Databricks](#option-4-gcp-databricks)
- [Workspace Configuration](#workspace-configuration)
- [Import Notebooks](#import-notebooks)
- [MLflow Setup](#mlflow-setup)
- [Troubleshooting](#troubleshooting)

---

## Option 1: Databricks Community Edition (Free)

**Best for**: Learning, prototyping, personal projects

### Step 1: Sign Up

1. Visit [community.cloud.databricks.com](https://community.cloud.databricks.com)
2. Click "Sign Up for Community Edition"
3. Fill in your details:
   - Email address
   - First and Last name
   - Create password
4. Verify your email
5. Accept terms and conditions

### Step 2: Access Your Workspace

1. Log in at [community.cloud.databricks.com](https://community.cloud.databricks.com)
2. You'll land on your workspace home page
3. Default cluster will be automatically created

### Limitations of Community Edition

- **Cluster**: Limited to 15GB RAM, single node
- **Runtime**: Auto-terminates after 2 hours of inactivity
- **Storage**: 10GB DBFS storage
- **Users**: Single user only
- **MLflow**: Tracking available, but limited model registry features

### Community Edition Benefits

✅ Free forever  
✅ Full notebook experience  
✅ Built-in MLflow tracking  
✅ Great for learning and prototyping  

---

## Option 2: AWS Databricks

**Best for**: Production workloads on AWS

### Prerequisites

- AWS Account
- AWS CLI configured
- Credit card for AWS charges

### Step 1: Sign Up via AWS Marketplace

1. Go to [AWS Marketplace](https://aws.amazon.com/marketplace)
2. Search for "Databricks"
3. Select "Databricks Unified Analytics Platform"
4. Click "Continue to Subscribe"
5. Accept terms and configure

### Step 2: Launch Databricks

1. After subscription, click "Continue to Configuration"
2. Select your region (e.g., us-east-1)
3. Click "Continue to Launch"
4. Choose launch option: "Launch from Website"
5. Create your Databricks account

### Step 3: Configure AWS Integration

```bash
# Create S3 bucket for Databricks
aws s3 mb s3://databricks-fraud-detection-artifacts

# Create IAM role for Databricks
aws iam create-role \
  --role-name databricks-fraud-detection-role \
  --assume-role-policy-document file://trust-policy.json

# Attach policies
aws iam attach-role-policy \
  --role-name databricks-fraud-detection-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### Pricing Estimate (AWS)

| Component | Cost |
|-----------|------|
| Databricks Platform | ~$0.55/DBU |
| EC2 Instances | Variable (e.g., m5.xlarge ~$0.192/hr) |
| S3 Storage | ~$0.023/GB/month |
| Data Transfer | Variable |

**Example**: Running 24/7 with m5.xlarge cluster ≈ $300-500/month

---

## Option 3: Azure Databricks

**Best for**: Production workloads on Azure

### Step 1: Create via Azure Portal

1. Log in to [Azure Portal](https://portal.azure.com)
2. Click "Create a resource"
3. Search for "Azure Databricks"
4. Click "Create"

### Step 2: Configure Workspace

**Basics Tab:**
- Subscription: Choose your subscription
- Resource Group: Create new or select existing
- Workspace Name: `fraud-detection-workspace`
- Region: Choose closest region
- Pricing Tier: Standard or Premium (Premium for MLflow Model Registry)

**Networking Tab:**
- Virtual Network: Optional for enhanced security
- Public IP: Enable if needed for external access

**Advanced Tab:**
- Managed Resource Group: Auto-generated or custom
- Encryption: Enable if required

### Step 3: Create Workspace

1. Click "Review + Create"
2. Validate configuration
3. Click "Create"
4. Wait 5-10 minutes for deployment

### Step 4: Launch Workspace

1. Go to your resource group
2. Click on the Databricks workspace
3. Click "Launch Workspace"

### Pricing Estimate (Azure)

| Tier | Cost/DBU |
|------|----------|
| Standard | $0.40/DBU |
| Premium | $0.55/DBU |
| + Azure VM costs | Variable |

---

## Option 4: GCP Databricks

**Best for**: Production workloads on GCP

### Step 1: Sign Up

1. Visit [accounts.cloud.databricks.com/registration.html](https://accounts.cloud.databricks.com/registration.html)
2. Select "Google Cloud Platform"
3. Create account

### Step 2: Configure GCP Project

```bash
# Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable storage-api.googleapis.com

# Create GCS bucket
gsutil mb gs://fraud-detection-artifacts
```

### Step 3: Create Workspace

Follow the Databricks setup wizard to connect your GCP project.

---

## Workspace Configuration

### Create Cluster (All Platforms)

1. **Navigate**: Click "Compute" in left sidebar
2. **Create Cluster**: Click "Create Cluster"

#### Recommended Cluster Configuration:

```yaml
Cluster Name: fraud-detection-cluster

Cluster Mode: Standard

Databricks Runtime Version: 14.3 LTS ML
  - Includes: Apache Spark 3.5.0, Scala 2.12
  - ML runtime includes: scikit-learn, xgboost, mlflow

Node Type:
  - AWS: m5.xlarge (4 cores, 16GB RAM)
  - Azure: Standard_DS3_v2 (4 cores, 14GB RAM)
  - GCP: n1-standard-4 (4 cores, 15GB RAM)

Autoscaling:
  - Min Workers: 2
  - Max Workers: 8
  - Enable autoscaling: ✓

Auto Termination: 120 minutes

Advanced Options:
  - Spark Config:
      spark.sql.adaptive.enabled true
      spark.sql.adaptive.coalescePartitions.enabled true
```

3. **Click "Create Cluster"**
4. Wait 3-5 minutes for cluster to start

### Install Libraries

#### Option A: Using Cluster UI

1. Click on your cluster
2. Go to "Libraries" tab
3. Click "Install New"
4. Select "PyPI"
5. Install these packages one by one:
   ```
   mlflow==2.8.0
   scikit-learn==1.3.0
   xgboost==1.7.6
   lightgbm==4.0.0
   imbalanced-learn==0.11.0
   optuna==3.3.0
   ```

#### Option B: Using requirements.txt

1. Upload `requirements.txt` to DBFS:
   ```bash
   databricks fs cp requirements.txt dbfs:/FileStore/requirements.txt
   ```

2. Install via notebook:
   ```python
   %pip install -r /dbfs/FileStore/requirements.txt
   ```

---

## Import Notebooks

### Method 1: Using Databricks UI

1. Click "Workspace" in left sidebar
2. Navigate to your user folder: `/Users/your.email@example.com/`
3. Click dropdown arrow → "Import"
4. Select "URL" or "File"

**For GitHub:**
```
https://github.com/yourusername/mlflow-fintech-fraud-pipeline/tree/main/notebooks
```

5. Import all `.py` notebooks from the project

### Method 2: Using Databricks CLI

```bash
# Install Databricks CLI
pip install databricks-cli

# Configure
databricks configure --token

# Import notebooks
databricks workspace import_dir notebooks /Users/your.email@example.com/fraud-detection
```

### Method 3: Using Git Integration (Recommended)

1. Click "Repos" in left sidebar
2. Click "Add Repo"
3. Enter your Git URL:
   ```
   https://github.com/yourusername/mlflow-fintech-fraud-pipeline.git
   ```
4. Click "Create Repo"
5. Notebooks are now synced with Git!

---

## MLflow Setup

### Built-in MLflow Tracking

Databricks has MLflow built-in! No additional setup needed.

### Verify MLflow

Run this in a notebook:
```python
import mlflow

# Check MLflow version
print(f"MLflow version: {mlflow.__version__}")

# Set experiment
mlflow.set_experiment("/Users/your.email@example.com/fraud_detection_test")

# Test logging
with mlflow.start_run():
    mlflow.log_param("test_param", "test_value")
    mlflow.log_metric("test_metric", 1.0)

print("✓ MLflow is working!")
```

### Access MLflow UI

1. Click "Experiments" icon in left sidebar
2. Or navigate to: `https://your-workspace.cloud.databricks.com/#mlflow`

---

## Data Upload

### Download Dataset from Kaggle

#### Option A: Using Kaggle API (Recommended)

1. Get Kaggle credentials:
   - Go to [kaggle.com/settings](https://www.kaggle.com/settings)
   - Scroll to "API"
   - Click "Create New API Token"
   - Download `kaggle.json`

2. Upload to Databricks:
   ```python
   # In notebook
   dbutils.fs.mkdirs("dbfs:/FileStore/kaggle/")
   
   # Upload kaggle.json via UI to /FileStore/kaggle/
   ```

3. Download dataset:
   ```python
   import os
   import json
   
   # Set Kaggle credentials
   with open('/dbfs/FileStore/kaggle/kaggle.json', 'r') as f:
       creds = json.load(f)
   
   os.environ['KAGGLE_USERNAME'] = creds['username']
   os.environ['KAGGLE_KEY'] = creds['key']
   
   # Download
   !pip install kaggle
   !kaggle datasets download -d mlg-ulb/creditcardfraud -p /dbfs/FileStore/fraud_detection/raw/
   !unzip /dbfs/FileStore/fraud_detection/raw/creditcardfraud.zip -d /dbfs/FileStore/fraud_detection/raw/
   ```

#### Option B: Manual Upload

1. Download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. In Databricks, click "Data" in left sidebar
3. Click "Add Data" → "Upload File"
4. Select `creditcard.csv`
5. Upload to `/FileStore/fraud_detection/raw/`

---

## Secrets Management

Store sensitive credentials securely:

```bash
# Using Databricks CLI
databricks secrets create-scope --scope fraud-detection

# Add secrets
databricks secrets put --scope fraud-detection --key kaggle-username
databricks secrets put --scope fraud-detection --key kaggle-key
databricks secrets put --scope fraud-detection --key aws-access-key
```

**Use in notebooks:**
```python
kaggle_user = dbutils.secrets.get(scope="fraud-detection", key="kaggle-username")
kaggle_key = dbutils.secrets.get(scope="fraud-detection", key="kaggle-key")
```

---

## Troubleshooting

### Issue: Cluster won't start

**Solution:**
- Check quotas in your cloud account
- Verify IAM roles/permissions
- Try different instance types
- Check region availability

### Issue: MLflow tracking not working

**Solution:**
```python
# Set experiment explicitly
mlflow.set_experiment("/Users/your.email@example.com/fraud_detection")

# Verify tracking URI
print(mlflow.get_tracking_uri())  # Should show 'databricks'
```

### Issue: Libraries not installing

**Solution:**
```python
# Use cell magic
%pip install package-name

# Restart Python kernel
dbutils.library.restartPython()
```

### Issue: Out of memory errors

**Solution:**
- Increase cluster size
- Use `.repartition()` for large DataFrames
- Enable autoscaling
- Sample data for development

### Issue: DBFS path not found

**Solution:**
```python
# Use correct DBFS path format
correct_path = "/dbfs/FileStore/fraud_detection/data.csv"
not_this = "dbfs:/FileStore/fraud_detection/data.csv"  # This won't work in pandas

# For Spark
spark_path = "dbfs:/FileStore/fraud_detection/data.csv"
```

---

## Next Steps

1. ✅ **Databricks workspace created**
2. ✅ **Cluster running**
3. ✅ **Notebooks imported**
4. ✅ **MLflow configured**
5. ✅ **Data uploaded**

**Now you're ready to:**
- Run `01_data_ingestion_eda.py`
- Start building your fraud detection pipeline!
- Track experiments with MLflow
- Deploy models to production

---

## Additional Resources

- [Databricks Documentation](https://docs.databricks.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Databricks Academy (Free Courses)](https://academy.databricks.com/)
- [MLflow Getting Started](https://mlflow.org/docs/latest/quickstart.html)

---

## Support

Having issues? Check these resources:
- [Databricks Community Forums](https://community.databricks.com/)
- [Stack Overflow - databricks tag](https://stackoverflow.com/questions/tagged/databricks)
- [MLflow GitHub Issues](https://github.com/mlflow/mlflow/issues)

---

**Ready to build?** Proceed to [`README.md`](../README.md) for the full project workflow!
