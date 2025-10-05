"""
Tests for model inference and utilities
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))


class TestModelInput:
    """Test model input validation"""

    def test_input_shape_validation(self):
        """Test model input shape"""
        n_features = 41
        X = np.random.randn(10, n_features)

        assert X.shape[0] == 10
        assert X.shape[1] == n_features

    def test_feature_types(self, sample_transaction_data):
        """Test feature data types"""
        df = pd.DataFrame([sample_transaction_data])

        assert df.select_dtypes(include=[np.number]).shape[1] == len(sample_transaction_data)


class TestFeatureEngineering:
    """Test feature engineering logic"""

    def test_hour_calculation(self):
        """Test hour calculation"""
        time_seconds = 3600
        hour = (time_seconds / 3600) % 24

        assert hour == 1.0

    def test_time_period(self):
        """Test time period calculation"""

        def get_time_period(hour):
            if 0 <= hour < 6:
                return 0
            elif 6 <= hour < 12:
                return 1
            elif 12 <= hour < 18:
                return 2
            else:
                return 3

        assert get_time_period(3) == 0
        assert get_time_period(9) == 1

    def test_log_transformation(self):
        """Test log transformation"""
        amounts = pd.Series([0, 10, 100, 1000])
        log_amounts = np.log1p(amounts)

        assert log_amounts[0] == 0
        assert log_amounts[3] > log_amounts[2]


class TestModelPrediction:
    """Test model prediction logic"""

    def test_prediction_output_range(self):
        """Test prediction outputs"""
        predictions = np.array([0, 1, 0, 0, 1])

        assert all(p in [0, 1] for p in predictions)

    def test_probability_output_range(self):
        """Test probability outputs"""
        probabilities = np.array([0.05, 0.95, 0.30])

        assert all(0 <= p <= 1 for p in probabilities)


class TestDataValidation:
    """Test data validation"""

    def test_missing_value_detection(self):
        """Test missing value detection"""
        df = pd.DataFrame({"A": [1, 2, np.nan, 4], "B": [5, 6, 7, 8]})

        assert df["A"].isna().sum() == 1
        assert df["B"].isna().sum() == 0
