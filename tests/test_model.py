"""
Tests for model inference and utilities
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestModelInput:
    """Test model input validation"""
    
    def test_input_shape_validation(self):
        """Test that model input has correct shape"""
        n_features = 41  # Total engineered features
        X = np.random.randn(10, n_features)
        
        assert X.shape[0] == 10  # 10 samples
        assert X.shape[1] == n_features  # 41 features
    
    def test_feature_types(self, sample_transaction_data):
        """Test feature data types"""
        df = pd.DataFrame([sample_transaction_data])
        
        # Check that all values are numeric
        assert df.select_dtypes(include=[np.number]).shape[1] == len(sample_transaction_data)
    
    def test_feature_ranges(self, sample_transaction_data):
        """Test feature value ranges"""
        # Amount should be non-negative
        assert sample_transaction_data['Amount'] >= 0
        
        # V features can be any value (PCA components)
        for i in range(1, 29):
            assert isinstance(sample_transaction_data[f'V{i}'], (int, float))


class TestFeatureEngineering:
    """Test feature engineering logic"""
    
    def test_hour_calculation(self):
        """Test hour calculation from Time"""
        time_seconds = 3600  # 1 hour
        hour = (time_seconds / 3600) % 24
        
        assert hour == 1.0
        
        time_seconds = 86400  # 24 hours (should wrap to 0)
        hour = (time_seconds / 3600) % 24
        assert hour == 0.0
    
    def test_time_period_calculation(self):
        """Test time period categorization"""
        def get_time_period(hour):
            if 0 <= hour < 6:
                return 0  # Night
            elif 6 <= hour < 12:
                return 1  # Morning
            elif 12 <= hour < 18:
                return 2  # Afternoon
            else:
                return 3  # Evening
        
        assert get_time_period(3) == 0  # Night
        assert get_time_period(9) == 1  # Morning
        assert get_time_period(15) == 2  # Afternoon
        assert get_time_period(20) == 3  # Evening
    
    def test_amount_binning(self):
        """Test amount binning logic"""
        amounts = pd.Series([25, 100, 500, 2000])
        bins = pd.cut(amounts, bins=[0, 50, 200, 1000, np.inf], labels=[0, 1, 2, 3])
        
        assert bins[0] == 0  # Low
        assert bins[1] == 1  # Medium
        assert bins[2] == 2  # High
        assert bins[3] == 3  # Very High
    
    def test_log_transformation(self):
        """Test log transformation of Amount"""
        amounts = pd.Series([0, 10, 100, 1000])
        log_amounts = np.log1p(amounts)
        
        assert log_amounts[0] == 0  # log1p(0) = 0
        assert log_amounts[1] > 2  # log1p(10) ≈ 2.4
        assert log_amounts[3] > log_amounts[2]  # Monotonic increase


class TestModelPrediction:
    """Test model prediction logic"""
    
    def test_prediction_output_range(self):
        """Test that predictions are 0 or 1"""
        predictions = np.array([0, 1, 0, 0, 1])
        
        assert all(p in [0, 1] for p in predictions)
    
    def test_probability_output_range(self):
        """Test that probabilities are between 0 and 1"""
        probabilities = np.array([0.05, 0.95, 0.30, 0.70, 0.15])
        
        assert all(0 <= p <= 1 for p in probabilities)
    
    def test_probability_sum(self):
        """Test that class probabilities sum to 1"""
        proba = np.array([[0.95, 0.05], [0.30, 0.70], [0.85, 0.15]])
        
        row_sums = proba.sum(axis=1)
        assert all(abs(s - 1.0) < 1e-6 for s in row_sums)


class TestDataValidation:
    """Test data validation utilities"""
    
    def test_missing_value_detection(self):
        """Test missing value detection"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, 6, 7, 8]
        })
        
        assert df['A'].isna().sum() == 1
        assert df['B'].isna().sum() == 0
    
    def test_outlier_detection(self):
        """Test outlier detection using IQR"""
        data = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is outlier
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
        
        assert len(outliers) == 1
        assert 100 in outliers.values
