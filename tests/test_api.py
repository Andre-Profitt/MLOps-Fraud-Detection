"""
Comprehensive tests for FastAPI application
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock all MLflow imports before importing app
mock_model = MagicMock()
mock_model.predict.return_value = np.array([0])
mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])

mock_scaler = MagicMock()
mock_scaler.transform.return_value = np.random.randn(1, 41)

feature_names = [f'V{i}' for i in range(1, 29)] + [
    'Amount', 'Hour', 'Time_Period', 'Amount_Bin', 'Amount_Log',
    'Amount_Per_Hour', 'Day_Of_Week', 'Is_Weekend',
    'V17_V14', 'V12_V10', 'V17_V12',
    'V17_squared', 'V14_squared', 'V12_squared'
]

with patch('mlflow.set_tracking_uri'), \
     patch('mlflow.sklearn.load_model', return_value=mock_model), \
     patch('builtins.open', create=True), \
     patch('pickle.load', side_effect=[mock_scaler, feature_names]):
    
    # Set global variables before import
    import src.api.main as main_module
    main_module.champion_model = mock_model
    main_module.challenger_model = mock_model
    main_module.scaler = mock_scaler
    main_module.feature_names = feature_names
    main_module.champion_version = "test-champion"
    main_module.challenger_version = "test-challenger"
    
    from src.api.main import app

client = TestClient(app)


class TestRootEndpoints:
    """Test root and health endpoints"""
    
    def test_read_root(self):
        """Test root endpoint returns service info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestPredictionEndpoint:
    """Test prediction endpoint"""
    
    def test_predict_valid_transaction(self, sample_transaction_data):
        """Test prediction with valid transaction data"""
        request_data = {
            "transaction_id": "test_123",
            "features": sample_transaction_data,
            "return_probability": True
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "fraud_probability" in data
        assert "model_version" in data
    
    def test_predict_missing_field(self):
        """Test prediction with missing fields"""
        incomplete_data = {
            "features": {
                "V1": -1.359807,
                "Amount": 149.62
            }
        }
        
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Test metrics endpoint"""
    
    def test_get_metrics(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "champion_version" in data
