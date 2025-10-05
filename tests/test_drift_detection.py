"""
Comprehensive tests for drift detection module
"""
import pytest
import numpy as np
import pandas as pd
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.monitoring.drift_detector import DriftDetector, create_baseline_from_data


class TestDriftDetector:
    """Test DriftDetector class"""
    
    def test_initialization_with_data(self, sample_dataframe):
        """Test DriftDetector initialization with baseline data"""
        detector = DriftDetector(baseline_data=sample_dataframe)
        
        assert detector.baseline_data is not None
        assert detector.baseline_stats is not None
        assert 'feature_stats' in detector.baseline_stats
        assert detector.psi_threshold == 0.2
        assert detector.ks_threshold == 0.05
    
    def test_initialization_custom_thresholds(self, sample_dataframe):
        """Test DriftDetector with custom thresholds"""
        detector = DriftDetector(
            baseline_data=sample_dataframe,
            psi_threshold=0.15,
            ks_threshold=0.01
        )
        
        assert detector.psi_threshold == 0.15
        assert detector.ks_threshold == 0.01


class TestPSICalculation:
    """Test Population Stability Index calculation"""
    
    def test_psi_no_drift(self):
        """Test PSI with no drift (same distribution)"""
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)
        
        detector = DriftDetector(baseline_data=pd.DataFrame({'feature': baseline}))
        psi = detector.calculate_psi(baseline, current)
        
        assert isinstance(psi, float)
        assert psi >= 0
        assert psi < 0.1  # Should be low for same distribution
    
    def test_psi_with_drift(self):
        """Test PSI with drift (shifted distribution)"""
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(1.5, 1, 1000)  # Shifted mean
        
        detector = DriftDetector(baseline_data=pd.DataFrame({'feature': baseline}))
        psi = detector.calculate_psi(baseline, current)
        
        assert psi > 0.1  # Should detect drift


class TestKSTest:
    """Test Kolmogorov-Smirnov test"""
    
    def test_ks_same_distribution(self):
        """Test KS test with same distribution"""
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)
        
        detector = DriftDetector(baseline_data=pd.DataFrame({'feature': baseline}))
        ks_stat, p_value = detector.calculate_ks_statistic(baseline, current)
        
        assert 0 <= ks_stat <= 1
        assert 0 <= p_value <= 1
        assert p_value > 0.05  # Should not reject null hypothesis
    
    def test_ks_different_distribution(self):
        """Test KS test with different distributions"""
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(2, 1, 1000)  # Different mean
        
        detector = DriftDetector(baseline_data=pd.DataFrame({'feature': baseline}))
        ks_stat, p_value = detector.calculate_ks_statistic(baseline, current)
        
        assert ks_stat > 0.1  # Should detect difference
        assert p_value < 0.05  # Should reject null hypothesis


class TestFeatureDriftDetection:
    """Test feature drift detection"""
    
    def test_detect_no_drift(self, sample_dataframe):
        """Test drift detection when there is no drift"""
        detector = DriftDetector(baseline_data=sample_dataframe, psi_threshold=0.2)
        
        # Create current data with same distribution
        current_data = sample_dataframe.copy()
        
        drift_results = detector.detect_feature_drift(current_data)
        
        assert 'timestamp' in drift_results
        assert 'features' in drift_results
        assert 'overall_drift_detected' in drift_results
        assert 'drifted_features' in drift_results
    
    def test_detect_drift_present(self, sample_dataframe):
        """Test drift detection when drift is present"""
        detector = DriftDetector(baseline_data=sample_dataframe, psi_threshold=0.1)
        
        # Create current data with drift
        current_data = sample_dataframe.copy()
        current_data['V1'] = current_data['V1'] + 2.0  # Add significant drift
        current_data['Amount'] = current_data['Amount'] * 1.5
        
        drift_results = detector.detect_feature_drift(current_data)
        
        assert drift_results['overall_drift_detected'] is True
        assert len(drift_results['drifted_features']) > 0
        assert 'V1' in drift_results['drifted_features']
