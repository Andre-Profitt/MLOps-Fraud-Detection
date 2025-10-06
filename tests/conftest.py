"""Pytest configuration and fixtures"""
import pytest
import numpy as np
import pandas as pd
import sys
from unittest.mock import MagicMock

# Mock all external dependencies before any imports
sys.modules['mlflow'] = MagicMock()
sys.modules['mlflow.sklearn'] = MagicMock()
sys.modules['mlflow.xgboost'] = MagicMock()
sys.modules['mlflow.lightgbm'] = MagicMock()
sys.modules['prometheus_fastapi_instrumentator'] = MagicMock()
sys.modules['prometheus_client'] = MagicMock()

@pytest.fixture
def sample_fraud_data():
    """Generate sample fraud detection data"""
    np.random.seed(42)
    n_samples = 1000
    data = {
        'Time': np.random.uniform(0, 172800, n_samples),
        'Amount': np.random.lognormal(3, 1.5, n_samples),
        **{f'V{i}': np.random.randn(n_samples) for i in range(1, 29)},
        'Class': np.random.binomial(1, 0.1, n_samples)
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_transaction_dict():
    """Sample transaction as dictionary"""
    return {
        "Time": 12345.0,
        "Amount": 149.62,
        **{f"V{i}": float(np.random.randn()) for i in range(1, 29)}
    }

@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test"""
    np.random.seed(42)
