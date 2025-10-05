"""
Pytest configuration and shared fixtures
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def sample_transaction_data():
    """Sample transaction data for testing"""
    return {
        'Time': 100.0,
        'V1': -1.359807,
        'V2': -0.072781,
        'V3': 2.536347,
        'V4': 1.378155,
        'V5': -0.338321,
        'V6': 0.462388,
        'V7': 0.239599,
        'V8': 0.098698,
        'V9': 0.363787,
        'V10': 0.090794,
        'V11': -0.551600,
        'V12': -0.617801,
        'V13': -0.991390,
        'V14': -0.311169,
        'V15': 1.468177,
        'V16': -0.470401,
        'V17': 0.207971,
        'V18': 0.025791,
        'V19': 0.403993,
        'V20': 0.251412,
        'V21': -0.018307,
        'V22': 0.277838,
        'V23': -0.110474,
        'V24': 0.066928,
        'V25': 0.128539,
        'V26': -0.189115,
        'V27': 0.133558,
        'V28': -0.021053,
        'Amount': 149.62
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing"""
    np.random.seed(42)
    data = {
        'V1': np.random.randn(100),
        'V2': np.random.randn(100),
        'V3': np.random.randn(100),
        'Amount': np.random.uniform(0, 1000, 100),
        'Class': np.random.choice([0, 1], 100, p=[0.99, 0.01])
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_model():
    """Mock ML model for testing"""
    model = MagicMock()
    model.predict.return_value = np.array([0])
    model.predict_proba.return_value = np.array([[0.95, 0.05]])
    return model


@pytest.fixture
def mock_scaler():
    """Mock scaler for testing"""
    scaler = MagicMock()
    scaler.transform.return_value = np.random.randn(1, 41)
    return scaler


@pytest.fixture
def feature_names():
    """Feature names for testing"""
    return [f'V{i}' for i in range(1, 29)] + [
        'Amount', 'Hour', 'Time_Period', 'Amount_Bin', 'Amount_Log',
        'Amount_Per_Hour', 'Day_Of_Week', 'Is_Weekend',
        'V17_V14', 'V12_V10', 'V17_V12',
        'V17_squared', 'V14_squared', 'V12_squared'
    ]
