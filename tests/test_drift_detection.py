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
        """Test DriftDetector initialization"""
        detector = DriftDetector(baseline_data=sample_dataframe)
        
        assert detector.baseline_data is not None
        assert detector.baseline_stats is not None
        assert detector.psi_threshold == 0.2
    
    def test_compute_baseline_stats(self, sample_dataframe):
        """Test baseline statistics computation"""
        detector = DriftDetector(baseline_data=sample_dataframe)
        stats = detector.baseline_stats
        
        assert 'timestamp' in stats
        assert 'n_samples' in stats
        assert stats['n_samples'] == len(sample_dataframe)


class TestPSICalculation:
    """Test PSI calculation"""
    
    def test_psi_no_drift(self):
        """Test PSI with no drift"""
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)
        
        detector = DriftDetector(baseline_data=pd.DataFrame({'feature': baseline}))
        psi = detector.calculate_psi(baseline, current)
        
        assert isinstance(psi, float)
        assert psi >= 0
        assert psi < 0.1
    
    def test_psi_with_drift(self):
        """Test PSI with drift"""
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(1.5, 1, 1000)
        
        detector = DriftDetector(baseline_data=pd.DataFrame({'feature': baseline}))
        psi = detector.calculate_psi(baseline, current)
        
        assert psi > 0.1


class TestKSTest:
    """Test KS test"""
    
    def test_ks_same_distribution(self):
        """Test KS with same distribution"""
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)
        
        detector = DriftDetector(baseline_data=pd.DataFrame({'feature': baseline}))
        ks_stat, p_value = detector.calculate_ks_statistic(baseline, current)
        
        assert 0 <= ks_stat <= 1
        assert 0 <= p_value <= 1


class TestFeatureDrift:
    """Test feature drift detection"""
    
    def test_detect_no_drift(self, sample_dataframe):
        """Test no drift detection"""
        detector = DriftDetector(baseline_data=sample_dataframe)
        current_data = sample_dataframe.copy()
        
        drift_results = detector.detect_feature_drift(current_data)
        
        assert 'features' in drift_results
        assert 'overall_drift_detected' in drift_results
    
    def test_detect_drift_present(self, sample_dataframe):
        """Test drift detection"""
        detector = DriftDetector(baseline_data=sample_dataframe, psi_threshold=0.1)
        
        current_data = sample_dataframe.copy()
        current_data['V1'] = current_data['V1'] + 2.0
        
        drift_results = detector.detect_feature_drift(current_data)
        
        assert drift_results['overall_drift_detected'] is True
        assert len(drift_results['drifted_features']) > 0
