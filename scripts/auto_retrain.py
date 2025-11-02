import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import logging
import warnings
import time
from typing import Dict, List, Any, Optional, Callable

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, model_path: str, data_path: str):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.project_root = Path(__file__).resolve().parent.parent
        self.monitoring_data = self.project_root / "models" / "monitoring_data.pkl"

        # Performance thresholds
        self.performance_thresholds = {
            "accuracy": 0.75,
            "f1_score": 0.70,
            "precision": 0.70,
            "recall": 0.70
        }

        # Data drift thresholds
        self.drift_thresholds = {
            "feature_drift": 0.1,
            "prediction_drift": 0.15
        }

    def load_model(self):
        """Load the current model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        return joblib.load(self.model_path)

    def check_performance(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Check model performance on test data"""
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1_score": f1_score(y_test, y_pred, average='weighted'),
                "timestamp": datetime.now().isoformat()
            }

            logger.info("Performance metrics: %s", metrics)
            return metrics

        except Exception as e:
            logger.error("Error checking performance: %s", str(e))
            return {}

    def detect_data_drift(self, current_data: pd.DataFrame, reference_data: pd.DataFrame) -> Dict:
        """Detect data drift between current and reference data"""
        try:
            drift_scores = {}

            # Simple statistical drift detection
            for column in current_data.select_dtypes(include=[np.number]).columns:
                if column in reference_data.columns:
                    current_mean = current_data[column].mean()
                    ref_mean = reference_data[column].mean()
                    current_std = current_data[column].std()
                    ref_std = reference_data[column].std()

                    # Calculate drift score (simplified)
                    if current_std > 0 and ref_std > 0:
                        drift_score = abs(current_mean - ref_mean) / ((current_std + ref_std) / 2)
                    else:
                        drift_score = abs(current_mean - ref_mean)

                    drift_scores[column] = drift_score

            avg_drift = np.mean(list(drift_scores.values())) if drift_scores else 0

            drift_result = {
                "feature_drift_scores": drift_scores,
                "average_drift": avg_drift,
                "drift_detected": avg_drift > self.drift_thresholds["feature_drift"],
                "timestamp": datetime.now().isoformat()
            }

            logger.info("Data drift analysis: %s", drift_result)
            return drift_result

        except Exception as e:
            logger.error("Error detecting data drift: %s", str(e))
            return {"error": str(e)}

    def should_retrain(self, current_metrics: Dict, drift_analysis: Dict) -> bool:
        """Determine if model should be retrained"""
        reasons = []

        # Check performance degradation
        for metric, threshold in self.performance_thresholds.items():
            if metric in current_metrics:
                if current_metrics[metric] < threshold:
                    reasons.append(f"Performance degradation in {metric}: {current_metrics[metric]:.3f} < {threshold}")

        # Check data drift
        if drift_analysis.get("drift_detected", False):
            reasons.append(f"Data drift detected: {drift_analysis['average_drift']:.3f}")

        if reasons:
            logger.info("Retrain recommended. Reasons: %s", reasons)
            return True, reasons

        return False, []

class AutomatedRetrainingPipeline:
    def __init__(self, model_path: str, data_path: str, training_function: Callable):
        self.model_path = model_path
        self.data_path = data_path
        self.training_function = training_function
        self.monitor = ModelMonitor(model_path, data_path)

        self.project_root = Path(__file__).resolve().parent.parent
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Retraining configuration
        self.retrain_interval_hours = 24  # Check every 24 hours
        self.last_retrain_check = None

    def load_latest_data(self) -> pd.DataFrame:
        """Load the latest dataset"""
        try:
            df = pd.read_csv(self.data_path)
            logger.info("Loaded dataset with shape: %s", df.shape)
            return df
        except Exception as e:
            logger.error("Error loading data: %s", str(e))
            return None

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2):
        """Split data into train and test sets"""
        try:
            from sklearn.model_selection import train_test_split

            # Prepare features and target
            feature_columns = ['flux', 'month', 'day', 'hour', 'day_of_year']
            X = df[feature_columns]
            y = df['flare_class_num']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error("Error splitting data: %s", str(e))
            return None, None, None, None

    def run_retraining_check(self) -> Dict:
        """Run automated retraining check"""
        logger.info("Running automated retraining check...")

        # Check if enough time has passed since last check
        if self.last_retrain_check:
            time_since_check = datetime.now() - self.last_retrain_check
            if time_since_check.total_seconds() < self.retrain_interval_hours * 3600:
                return {"status": "skipped", "reason": "Too soon since last check"}

        self.last_retrain_check = datetime.now()

        # Load data
        df = self.load_latest_data()
        if df is None:
            return {"status": "failed", "reason": "Could not load data"}

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df)
        if X_test is None:
            return {"status": "failed", "reason": "Could not split data"}

        # Load current model
        try:
            model = self.monitor.load_model()
        except Exception as e:
            return {"status": "failed", "reason": f"Could not load model: {str(e)}"}

        # Check performance
        current_metrics = self.monitor.check_performance(model, X_test, y_test)

        # Check data drift (using training data as reference)
        drift_analysis = self.monitor.detect_data_drift(X_test, X_train)

        # Determine if retraining is needed
        should_retrain, reasons = self.monitor.should_retrain(current_metrics, drift_analysis)

        if should_retrain:
            logger.info("Starting automated retraining...")
            retrain_result = self.perform_retraining(df)

            return {
                "status": "retrained",
                "retrain_result": retrain_result,
                "performance_metrics": current_metrics,
                "drift_analysis": drift_analysis,
                "retrain_reasons": reasons
            }

        return {
            "status": "no_action",
            "performance_metrics": current_metrics,
            "drift_analysis": drift_analysis,
            "message": "Model performance is acceptable"
        }

    def perform_retraining(self, df: pd.DataFrame) -> Dict:
        """Perform model retraining"""
        try:
            logger.info("Starting model retraining...")

            # Split data
            X_train, X_test, y_train, y_test = self.split_data(df)
            if X_test is None:
                return {"status": "failed", "reason": "Could not split data for retraining"}

            # Train new model
            new_model, metrics = self.training_function(X_train, y_train, X_test, y_test)

            # Save new model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.model_path.parent / f"model_backup_{timestamp}.joblib"
            new_model_path = self.model_path.parent / f"model_retrained_{timestamp}.joblib"

            # Backup current model
            if self.model_path.exists():
                import shutil
                shutil.copy2(self.model_path, backup_path)
                logger.info("Backed up current model to: %s", backup_path)

            # Save new model
            joblib.dump(new_model, new_model_path)
            logger.info("Saved new model to: %s", new_model_path)

            # Replace current model
            joblib.dump(new_model, self.model_path)
            logger.info("Updated current model: %s", self.model_path)

            return {
                "status": "success",
                "new_model_path": str(new_model_path),
                "backup_path": str(backup_path),
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("Error during retraining: %s", str(e))
            return {"status": "failed", "reason": str(e)}

def demo_training_function(X_train, y_train, X_test, y_test):
    """Demo training function for testing"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    logger.info("Training demo model...")

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted')
    }

    logger.info("Demo model trained with metrics: %s", metrics)
    return model, metrics

def run_automated_retraining_demo():
    """Demonstrate automated retraining pipeline"""
    logger.info("🚀 Starting Automated Retraining Demo")

    # Initialize pipeline
    pipeline = AutomatedRetrainingPipeline(
        model_path="models/model_rf_improved.joblib",
        data_path="data/historical_goes_2010_2015_parsed.csv",
        training_function=demo_training_function
    )

    # Run retraining check
    result = pipeline.run_retraining_check()

    logger.info("Retraining check result: %s", result)
    logger.info("✅ Automated retraining demo complete!")

if __name__ == "__main__":
    run_automated_retraining_demo()
