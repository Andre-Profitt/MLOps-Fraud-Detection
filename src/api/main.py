"""
FastAPI application for fraud detection inference
Supports real-time predictions with MLflow model serving
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import pickle
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection using MLflow models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
prediction_counter = Counter(
    'prediction_requests_total',
    'Total prediction requests',
    ['model_version', 'prediction']
)

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_version']
)

fraud_predictions = Counter(
    'fraud_predictions_total',
    'Total fraud predictions',
    ['model_version']
)

model_confidence = Histogram(
    'model_confidence_score',
    'Model confidence scores',
    ['model_version', 'prediction']
)

# Initialize Prometheus instrumentator
instrumentator = Instrumentator().instrument(app)

# Global variables for models
champion_model = None
challenger_model = None
scaler = None
feature_names = None
champion_version = "unknown"
challenger_version = "unknown"

# Model configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "databricks")
CHAMPION_MODEL_URI = os.getenv("CHAMPION_MODEL_URI", "models:/fraud_detection_champion/Production")
CHALLENGER_MODEL_URI = os.getenv("CHALLENGER_MODEL_URI", "models:/fraud_detection_challenger/Staging")
SCALER_PATH = os.getenv("SCALER_PATH", "/dbfs/FileStore/fraud_detection/prepared/scaler.pkl")
FEATURE_NAMES_PATH = os.getenv("FEATURE_NAMES_PATH", "/dbfs/FileStore/fraud_detection/prepared/feature_names.pkl")

# A/B testing configuration
AB_TEST_ENABLED = os.getenv("AB_TEST_ENABLED", "true").lower() == "true"
CHALLENGER_TRAFFIC_PERCENTAGE = float(os.getenv("CHALLENGER_TRAFFIC_PERCENTAGE", "10"))


class TransactionFeatures(BaseModel):
    """Transaction features for prediction"""
    Time: Optional[float] = Field(None, description="Seconds elapsed between this transaction and the first")
    V1: float = Field(..., description="PCA component 1")
    V2: float = Field(..., description="PCA component 2")
    V3: float = Field(..., description="PCA component 3")
    V4: float = Field(..., description="PCA component 4")
    V5: float = Field(..., description="PCA component 5")
    V6: float = Field(..., description="PCA component 6")
    V7: float = Field(..., description="PCA component 7")
    V8: float = Field(..., description="PCA component 8")
    V9: float = Field(..., description="PCA component 9")
    V10: float = Field(..., description="PCA component 10")
    V11: float = Field(..., description="PCA component 11")
    V12: float = Field(..., description="PCA component 12")
    V13: float = Field(..., description="PCA component 13")
    V14: float = Field(..., description="PCA component 14")
    V15: float = Field(..., description="PCA component 15")
    V16: float = Field(..., description="PCA component 16")
    V17: float = Field(..., description="PCA component 17")
    V18: float = Field(..., description="PCA component 18")
    V19: float = Field(..., description="PCA component 19")
    V20: float = Field(..., description="PCA component 20")
    V21: float = Field(..., description="PCA component 21")
    V22: float = Field(..., description="PCA component 22")
    V23: float = Field(..., description="PCA component 23")
    V24: float = Field(..., description="PCA component 24")
    V25: float = Field(..., description="PCA component 25")
    V26: float = Field(..., description="PCA component 26")
    V27: float = Field(..., description="PCA component 27")
    V28: float = Field(..., description="PCA component 28")
    Amount: float = Field(..., ge=0, description="Transaction amount")
    
    class Config:
        schema_extra = {
            "example": {
                "Time": 0.0,
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.025791,
                "V19": 0.403993,
                "V20": 0.251412,
                "V21": -0.018307,
                "V22": 0.277838,
                "V23": -0.110474,
                "V24": 0.066928,
                "V25": 0.128539,
                "V26": -0.189115,
                "V27": 0.133558,
                "V28": -0.021053,
                "Amount": 149.62
            }
        }


class PredictionRequest(BaseModel):
    """Prediction request with transaction features"""
    transaction_id: Optional[str] = Field(None, description="Unique transaction identifier")
    features: TransactionFeatures
    return_probability: bool = Field(True, description="Return fraud probability")


class PredictionResponse(BaseModel):
    """Prediction response"""
    transaction_id: Optional[str]
    prediction: int = Field(..., description="0 = Normal, 1 = Fraud")
    fraud_probability: Optional[float] = Field(None, ge=0, le=1, description="Probability of fraud")
    model_version: str
    timestamp: datetime
    processing_time_ms: float


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    transactions: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_transactions: int
    fraud_count: int
    processing_time_ms: float


def load_models():
    """Load MLflow models and preprocessing artifacts"""
    global champion_model, challenger_model, scaler, feature_names
    global champion_version, challenger_version
    
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Load champion model
        logger.info(f"Loading champion model from: {CHAMPION_MODEL_URI}")
        champion_model = mlflow.sklearn.load_model(CHAMPION_MODEL_URI)
        champion_version = "production"
        logger.info("Champion model loaded successfully")
        
        # Load challenger model if A/B testing enabled
        if AB_TEST_ENABLED:
            try:
                logger.info(f"Loading challenger model from: {CHALLENGER_MODEL_URI}")
                challenger_model = mlflow.sklearn.load_model(CHALLENGER_MODEL_URI)
                challenger_version = "staging"
                logger.info("Challenger model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load challenger model: {e}")
                challenger_model = None
        
        # Load scaler
        logger.info(f"Loading scaler from: {SCALER_PATH}")
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded successfully")
        
        # Load feature names
        logger.info(f"Loading feature names from: {FEATURE_NAMES_PATH}")
        with open(FEATURE_NAMES_PATH, 'rb') as f:
            feature_names = pickle.load(f)
        logger.info(f"Feature names loaded: {len(feature_names)} features")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


def engineer_features(features_dict: Dict) -> pd.DataFrame:
    """
    Engineer features from raw transaction data
    """
    df = pd.DataFrame([features_dict])
    
    # Feature engineering (matching training notebook)
    if 'Time' in df.columns and df['Time'].notna().all():
        df['Hour'] = (df['Time'] / 3600) % 24
    else:
        df['Hour'] = 0
    
    # Time period
    def get_time_period(hour):
        if 0 <= hour < 6:
            return 0  # Night
        elif 6 <= hour < 12:
            return 1  # Morning
        elif 12 <= hour < 18:
            return 2  # Afternoon
        else:
            return 3  # Evening
    
    df['Time_Period'] = df['Hour'].apply(get_time_period)
    
    # Amount bins
    df['Amount_Bin'] = pd.cut(df['Amount'], 
                              bins=[0, 50, 200, 1000, np.inf],
                              labels=[0, 1, 2, 3]).astype(int)
    
    # Log transformation
    df['Amount_Log'] = np.log1p(df['Amount'])
    
    # Amount per hour
    df['Amount_Per_Hour'] = df['Amount'] / (df['Hour'] + 1)
    
    # Day of week
    if 'Time' in df.columns and df['Time'].notna().all():
        df['Day_Of_Week'] = ((df['Time'] // 86400) % 7).astype(int)
        df['Is_Weekend'] = (df['Day_Of_Week'] >= 5).astype(int)
    else:
        df['Day_Of_Week'] = 0
        df['Is_Weekend'] = 0
    
    # V feature interactions
    df['V17_V14'] = df['V17'] * df['V14']
    df['V12_V10'] = df['V12'] * df['V10']
    df['V17_V12'] = df['V17'] * df['V12']
    
    # Polynomial features
    df['V17_squared'] = df['V17'] ** 2
    df['V14_squared'] = df['V14'] ** 2
    df['V12_squared'] = df['V12'] ** 2
    
    # Drop Time column
    if 'Time' in df.columns:
        df = df.drop('Time', axis=1)
    
    return df


def select_model(use_ab_test: bool = True) -> tuple:
    """
    Select model based on A/B testing configuration
    Returns: (model, model_version)
    """
    if not use_ab_test or challenger_model is None:
        return champion_model, champion_version
    
    # A/B test: random traffic split
    import random
    if random.random() * 100 < CHALLENGER_TRAFFIC_PERCENTAGE:
        return challenger_model, challenger_version
    else:
        return champion_model, champion_version


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting Fraud Detection API...")
    load_models()
    logger.info("API ready to serve predictions")
    instrumentator.expose(app)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "status": "healthy",
        "models": {
            "champion": champion_version,
            "challenger": challenger_version if challenger_model else "not loaded"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": champion_model is not None,
        "ab_test_enabled": AB_TEST_ENABLED
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Single transaction prediction
    """
    start_time = datetime.now()
    
    try:
        # Select model (A/B testing)
        model, model_version = select_model(use_ab_test=AB_TEST_ENABLED)
        
        # Convert features to dict
        features_dict = request.features.dict()
        
        # Engineer features
        df_features = engineer_features(features_dict)
        
        # Ensure correct feature order
        df_features = df_features[feature_names]
        
        # Scale features
        features_scaled = scaler.transform(df_features)
        
        # Predict
        with prediction_latency.labels(model_version=model_version).time():
            prediction = int(model.predict(features_scaled)[0])
            fraud_prob = float(model.predict_proba(features_scaled)[0][1]) if request.return_probability else None
        
        # Update metrics
        prediction_counter.labels(
            model_version=model_version,
            prediction='fraud' if prediction == 1 else 'normal'
        ).inc()
        
        if prediction == 1:
            fraud_predictions.labels(model_version=model_version).inc()
        
        if fraud_prob is not None:
            model_confidence.labels(
                model_version=model_version,
                prediction='fraud' if prediction == 1 else 'normal'
            ).observe(fraud_prob if prediction == 1 else 1 - fraud_prob)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PredictionResponse(
            transaction_id=request.transaction_id,
            prediction=prediction,
            fraud_probability=fraud_prob,
            model_version=model_version,
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch transaction predictions
    """
    start_time = datetime.now()
    
    try:
        predictions = []
        fraud_count = 0
        
        for transaction in request.transactions:
            pred_response = await predict(transaction)
            predictions.append(pred_response)
            if pred_response.prediction == 1:
                fraud_count += 1
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_transactions=len(predictions),
            fraud_count=fraud_count,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Get prediction metrics"""
    return {
        "champion_version": champion_version,
        "challenger_version": challenger_version if challenger_model else None,
        "ab_test_enabled": AB_TEST_ENABLED,
        "challenger_traffic_percentage": CHALLENGER_TRAFFIC_PERCENTAGE if AB_TEST_ENABLED else 0
    }


@app.post("/reload")
async def reload_models():
    """
    Reload models from MLflow (for updates)
    Requires authentication in production
    """
    try:
        load_models()
        return {
            "status": "success",
            "message": "Models reloaded successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
