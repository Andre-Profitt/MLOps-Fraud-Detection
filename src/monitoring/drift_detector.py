"""
Drift Detection Module
Implements data drift and model drift detection using:
- Population Stability Index (PSI)
- Kolmogorov-Smirnov (KS) Test
- Model performance monitoring
"""

import json
import logging
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Comprehensive drift detection for fraud detection model
    """

    def __init__(
        self,
        baseline_data: pd.DataFrame = None,
        baseline_path: str = None,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
    ):
        """
        Initialize drift detector

        Args:
            baseline_data: Reference dataset for comparison
            baseline_path: Path to saved baseline data
            psi_threshold: PSI threshold (0.1 = small shift, 0.2 = moderate, 0.25+ = large)
            ks_threshold: KS test p-value threshold
        """
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold

        if baseline_data is not None:
            self.baseline_data = baseline_data
            self.baseline_stats = self._compute_baseline_stats(baseline_data)
        elif baseline_path is not None:
            self._load_baseline(baseline_path)
        else:
            self.baseline_data = None
            self.baseline_stats = None

    def _compute_baseline_stats(self, data: pd.DataFrame) -> Dict:
        """Compute baseline statistics"""
        stats = {"feature_stats": {}, "timestamp": datetime.now().isoformat(), "n_samples": len(data)}

        for col in data.columns:
            if data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                stats["feature_stats"][col] = {
                    "mean": float(data[col].mean()),
                    "std": float(data[col].std()),
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                    "q25": float(data[col].quantile(0.25)),
                    "q50": float(data[col].quantile(0.50)),
                    "q75": float(data[col].quantile(0.75)),
                    "distribution": data[col].values.tolist()[:10000],  # Sample for KS test
                }

        return stats

    def save_baseline(self, path: str):
        """Save baseline statistics"""
        if self.baseline_stats is None:
            raise ValueError("No baseline data available to save")

        with open(path, "w") as f:
            # Don't save full distribution arrays in JSON
            stats_to_save = {
                "feature_stats": {
                    k: {sk: sv for sk, sv in v.items() if sk != "distribution"}
                    for k, v in self.baseline_stats["feature_stats"].items()
                },
                "timestamp": self.baseline_stats["timestamp"],
                "n_samples": self.baseline_stats["n_samples"],
            }
            json.dump(stats_to_save, f, indent=2)

        # Save full baseline data separately
        baseline_data_path = path.replace(".json", "_data.pkl")
        with open(baseline_data_path, "wb") as f:
            pickle.dump(self.baseline_data, f)

        logger.info(f"Baseline saved to {path}")

    def _load_baseline(self, path: str):
        """Load baseline statistics"""
        with open(path, "r") as f:
            self.baseline_stats = json.load(f)

        # Load full baseline data
        baseline_data_path = path.replace(".json", "_data.pkl")
        with open(baseline_data_path, "rb") as f:
            self.baseline_data = pickle.load(f)

        logger.info(f"Baseline loaded from {path}")

    def calculate_psi(self, baseline_array: np.ndarray, current_array: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)

        PSI Interpretation:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Small change
        - PSI >= 0.2: Significant change (requires action)

        Args:
            baseline_array: Baseline feature values
            current_array: Current feature values
            bins: Number of bins for discretization

        Returns:
            PSI value
        """
        # Define bins based on baseline
        min_val = min(baseline_array.min(), current_array.min())
        max_val = max(baseline_array.max(), current_array.max())

        breakpoints = np.linspace(min_val, max_val, bins + 1)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        # Calculate distributions
        baseline_dist = np.histogram(baseline_array, bins=breakpoints)[0]
        current_dist = np.histogram(current_array, bins=breakpoints)[0]

        # Convert to percentages
        baseline_pct = baseline_dist / len(baseline_array)
        current_pct = current_dist / len(current_array)

        # Avoid division by zero
        baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
        current_pct = np.where(current_pct == 0, 0.0001, current_pct)

        # Calculate PSI
        psi_values = (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
        psi = np.sum(psi_values)

        return float(psi)

    def calculate_ks_statistic(self, baseline_array: np.ndarray, current_array: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Kolmogorov-Smirnov test statistic

        Args:
            baseline_array: Baseline feature values
            current_array: Current feature values

        Returns:
            Tuple of (KS statistic, p-value)
        """
        ks_statistic, p_value = stats.ks_2samp(baseline_array, current_array)
        return float(ks_statistic), float(p_value)

    def detect_feature_drift(self, current_data: pd.DataFrame, features: Optional[List[str]] = None) -> Dict:
        """
        Detect drift in features

        Args:
            current_data: Current dataset
            features: List of features to check (None = all features)

        Returns:
            Dictionary with drift results per feature
        """
        if self.baseline_data is None:
            raise ValueError("No baseline data available")

        if features is None:
            features = [col for col in self.baseline_data.columns if col in current_data.columns]

        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "features": {},
            "overall_drift_detected": False,
            "drifted_features": [],
        }

        for feature in features:
            if feature not in self.baseline_data.columns or feature not in current_data.columns:
                continue

            baseline_values = self.baseline_data[feature].dropna().values
            current_values = current_data[feature].dropna().values

            # Calculate PSI
            psi = self.calculate_psi(baseline_values, current_values)

            # Calculate KS test
            ks_stat, p_value = self.calculate_ks_statistic(baseline_values, current_values)

            # Determine drift
            psi_drift = psi >= self.psi_threshold
            ks_drift = p_value < self.ks_threshold
            drift_detected = psi_drift or ks_drift

            if drift_detected:
                drift_results["drifted_features"].append(feature)
                drift_results["overall_drift_detected"] = True

            drift_results["features"][feature] = {
                "psi": psi,
                "psi_threshold": self.psi_threshold,
                "psi_drift": psi_drift,
                "ks_statistic": ks_stat,
                "ks_p_value": p_value,
                "ks_threshold": self.ks_threshold,
                "ks_drift": ks_drift,
                "drift_detected": drift_detected,
                "baseline_mean": float(baseline_values.mean()),
                "current_mean": float(current_values.mean()),
                "baseline_std": float(baseline_values.std()),
                "current_std": float(current_values.std()),
            }

        return drift_results

    def detect_prediction_drift(
        self,
        baseline_predictions: np.ndarray,
        current_predictions: np.ndarray,
        baseline_probabilities: Optional[np.ndarray] = None,
        current_probabilities: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Detect drift in model predictions

        Args:
            baseline_predictions: Baseline predictions (0/1)
            current_predictions: Current predictions (0/1)
            baseline_probabilities: Baseline prediction probabilities
            current_probabilities: Current prediction probabilities

        Returns:
            Dictionary with prediction drift results
        """
        drift_results = {"timestamp": datetime.now().isoformat(), "prediction_drift": {}}

        # Drift in prediction distribution
        baseline_fraud_rate = baseline_predictions.mean()
        current_fraud_rate = current_predictions.mean()

        # Chi-square test for prediction distribution
        baseline_counts = np.bincount(baseline_predictions.astype(int), minlength=2)
        current_counts = np.bincount(current_predictions.astype(int), minlength=2)

        chi2_stat, chi2_p_value = stats.chisquare(
            current_counts, f_exp=baseline_counts * (len(current_predictions) / len(baseline_predictions))
        )

        drift_results["prediction_drift"] = {
            "baseline_fraud_rate": float(baseline_fraud_rate),
            "current_fraud_rate": float(current_fraud_rate),
            "fraud_rate_change": float(current_fraud_rate - baseline_fraud_rate),
            "chi2_statistic": float(chi2_stat),
            "chi2_p_value": float(chi2_p_value),
            "drift_detected": chi2_p_value < 0.05,
        }

        # Drift in prediction probabilities (if available)
        if baseline_probabilities is not None and current_probabilities is not None:
            psi_proba = self.calculate_psi(baseline_probabilities, current_probabilities)
            ks_stat, ks_p_value = self.calculate_ks_statistic(baseline_probabilities, current_probabilities)

            drift_results["probability_drift"] = {
                "psi": float(psi_proba),
                "psi_drift": psi_proba >= self.psi_threshold,
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(ks_p_value),
                "ks_drift": ks_p_value < self.ks_threshold,
                "drift_detected": psi_proba >= self.psi_threshold or ks_p_value < self.ks_threshold,
            }

        return drift_results

    def generate_drift_report(
        self,
        current_data: pd.DataFrame,
        current_predictions: Optional[np.ndarray] = None,
        current_probabilities: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Generate comprehensive drift report

        Args:
            current_data: Current dataset
            current_predictions: Current model predictions
            current_probabilities: Current prediction probabilities

        Returns:
            Complete drift report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "baseline_timestamp": self.baseline_stats["timestamp"],
            "baseline_samples": self.baseline_stats["n_samples"],
            "current_samples": len(current_data),
            "drift_summary": {},
        }

        # Feature drift
        feature_drift = self.detect_feature_drift(current_data)
        report["feature_drift"] = feature_drift
        report["drift_summary"]["feature_drift_detected"] = feature_drift["overall_drift_detected"]
        report["drift_summary"]["drifted_features_count"] = len(feature_drift["drifted_features"])
        report["drift_summary"]["drifted_features"] = feature_drift["drifted_features"]

        # Prediction drift (if predictions provided)
        if current_predictions is not None:
            # Need baseline predictions
            logger.warning("Baseline predictions not available for prediction drift")

        # Overall drift status
        report["drift_summary"]["overall_status"] = (
            "DRIFT_DETECTED" if feature_drift["overall_drift_detected"] else "NO_DRIFT"
        )

        # Recommendations
        if feature_drift["overall_drift_detected"]:
            report["drift_summary"]["recommendations"] = [
                "Data drift detected - consider retraining the model",
                f"Features with drift: {', '.join(feature_drift['drifted_features'][:5])}",
                "Monitor model performance closely",
                "Collect more recent data for model update",
            ]
        else:
            report["drift_summary"]["recommendations"] = ["No significant drift detected", "Continue monitoring"]

        return report

    def should_trigger_retraining(self, drift_report: Dict, max_drifted_features: int = 5) -> Tuple[bool, str]:
        """
        Determine if model retraining should be triggered

        Args:
            drift_report: Drift report from generate_drift_report
            max_drifted_features: Maximum number of drifted features to tolerate

        Returns:
            Tuple of (should_retrain, reason)
        """
        feature_drift = drift_report["feature_drift"]

        if not feature_drift["overall_drift_detected"]:
            return False, "No drift detected"

        drifted_count = len(feature_drift["drifted_features"])

        if drifted_count > max_drifted_features:
            return True, f"Too many features drifted ({drifted_count} > {max_drifted_features})"

        # Check for critical features with high PSI
        critical_drift = any(details["psi"] > 0.25 for details in feature_drift["features"].values())

        if critical_drift:
            return True, "Critical feature drift detected (PSI > 0.25)"

        return False, f"Drift within acceptable range ({drifted_count} features)"


def create_baseline_from_data(
    data: pd.DataFrame, output_path: str, psi_threshold: float = 0.2, ks_threshold: float = 0.05
):
    """
    Create and save baseline for drift detection

    Args:
        data: Baseline dataset
        output_path: Path to save baseline
        psi_threshold: PSI threshold
        ks_threshold: KS test threshold
    """
    detector = DriftDetector(baseline_data=data, psi_threshold=psi_threshold, ks_threshold=ks_threshold)

    detector.save_baseline(output_path)
    logger.info(f"Baseline created with {len(data)} samples")

    return detector


# Example usage
if __name__ == "__main__":
    # Example: Create baseline from training data
    import pandas as pd

    # Load your training data
    # df_train = pd.read_parquet('/path/to/training_data.parquet')
    # Create baseline
    # detector = create_baseline_from_data(
    #     data=df_train,
    #     output_path='/path/to/baseline.json'
    # )
    # Later: Detect drift on new data
    # df_new = pd.read_parquet('/path/to/new_data.parquet')
    # drift_report = detector.generate_drift_report(df_new)
    # should_retrain, reason = detector.should_trigger_retraining(drift_report)

    print("Drift detection module ready!")
