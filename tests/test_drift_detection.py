"""Tests for drift detection module"""
import pytest
import numpy as np
import pandas as pd
from src.monitoring.drift_detector import DriftDetector

class TestDriftDetector:
    """Test drift detection functionality"""
    
    def test_initialization(self, sample_fraud_data):
        """Test detector initialization"""
        detector = DriftDetector(baseline_data=sample_fraud_data.drop('Class', axis=1))
        assert detector.baseline_data is not None
        assert detector.psi_threshold == 0.2
    
    def test_psi_calculation_no_drift(self, sample_fraud_data):
        """Test PSI with no drift"""
        baseline = sample_fraud_data.drop('Class', axis=1)
        detector = DriftDetector(baseline_data=baseline)
        
        # Same distribution
        current = baseline.sample(500, random_state=43)
        psi = detector.calculate_psi(baseline['Amount'].values, current['Amount'].values)
        
        assert psi < 0.1, "PSI should be low for same distribution"
    
    def test_psi_calculation_with_drift(self, sample_fraud_data):
        """Test PSI with drift"""
        baseline = sample_fraud_data.drop('Class', axis=1)
        detector = DriftDetector(baseline_data=baseline)
        
        # Shifted distribution
        current = baseline.copy()
        current['Amount'] = current["Amount"] * 5  # Significant shift
        
        psi = detector.calculate_psi(baseline['Amount'].values, current['Amount'].values)
        
        assert psi > 0.2, "PSI should be high for shifted distribution"
    
    def test_ks_statistic(self, sample_fraud_data):
        """Test KS statistic calculation"""
        baseline = sample_fraud_data.drop('Class', axis=1)
        detector = DriftDetector(baseline_data=baseline)
        
        ks_stat, p_value = detector.calculate_ks_statistic(
            baseline['Amount'].values,
            baseline['Amount'].values
        )
        
        assert ks_stat == 0.0
        assert p_value == 1.0
    
    def test_feature_drift_detection(self, sample_fraud_data):
        """Test feature drift detection"""
        baseline = sample_fraud_data.drop('Class', axis=1)
        detector = DriftDetector(baseline_data=baseline)
        
        current = baseline.sample(500, random_state=45)
        results = detector.detect_feature_drift(current)
        
        assert 'features' in results
        assert 'overall_drift_detected' in results
        assert isinstance(results['overall_drift_detected'], bool)
    
    def test_drift_report_generation(self, sample_fraud_data):
        """Test drift report generation"""
        baseline = sample_fraud_data.drop('Class', axis=1)
        detector = DriftDetector(baseline_data=baseline)
        
        current = baseline.sample(500, random_state=46)
        report = detector.generate_drift_report(current)
        
        assert 'drift_summary' in report
        assert 'feature_drift' in report
        assert 'timestamp' in report
    
    def test_retraining_decision(self, sample_fraud_data):
        """Test retraining decision logic"""
        baseline = sample_fraud_data.drop('Class', axis=1)
        detector = DriftDetector(baseline_data=baseline)
        
        current = baseline.sample(500, random_state=47)
        report = detector.generate_drift_report(current)
        
        should_retrain, reason = detector.should_trigger_retraining(report)
        
        assert isinstance(should_retrain, bool)
        assert isinstance(reason, str)
