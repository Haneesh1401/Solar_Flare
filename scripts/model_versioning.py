import pandas as pd
import numpy as np
import joblib
import pickle
import json
from pathlib import Path
from datetime import datetime
import hashlib
import logging
from typing import Dict, List, Any, Optional
import warnings
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.registry_path = self.project_root / "models" / "model_registry.json"
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize registry
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load model registry from file"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except:
                logger.warning("Could not load registry, creating new one")
                return {}
        return {}

    def _save_registry(self):
        """Save registry to file"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def _generate_model_hash(self, model_params: Dict) -> str:
        """Generate hash for model parameters"""
        param_str = json.dumps(model_params, sort_keys=True)
        return hashlib.sha256(param_str.encode()).hexdigest()[:8]

    def register_model(self, model_name: str, model, model_params: Dict,
                      metrics: Dict, model_type: str = "sklearn") -> str:
        """Register a new model version"""

        # Generate model hash
        model_hash = self._generate_model_hash(model_params)

        # Create version info
        timestamp = datetime.now().isoformat()
        version = f"v_{len([k for k in self.registry.keys() if k.startswith(model_name)]) + 1}_{model_hash}"

        # Save model
        model_path = self.models_dir / f"{model_name}_{version}.joblib"
        if model_type == "sklearn":
            joblib.dump(model, model_path)
        else:
            # For TensorFlow/Keras models
            model.save(str(model_path))

        # Update registry
        self.registry[version] = {
            "model_name": model_name,
            "version": version,
            "model_path": str(model_path),
            "model_params": model_params,
            "metrics": metrics,
            "created_at": timestamp,
            "model_type": model_type,
            "model_hash": model_hash
        }

        self._save_registry()
        logger.info(f"Model {model_name} registered as {version}")
        return version

    def get_model(self, version: str):
        """Load model by version"""
        if version not in self.registry:
            raise ValueError(f"Model version {version} not found")

        model_info = self.registry[version]
        model_path = Path(model_info["model_path"])

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if model_info["model_type"] == "sklearn":
            return joblib.load(model_path)
        else:
            # For TensorFlow models
            import tensorflow as tf
            return tf.keras.models.load_model(str(model_path))

    def get_best_model(self, model_name: str, metric: str = "f1_score") -> Optional[str]:
        """Get best performing model version for a given model name"""
        model_versions = [v for v in self.registry.keys()
                         if self.registry[v]["model_name"] == model_name]

        if not model_versions:
            return None

        best_version = max(model_versions,
                          key=lambda v: self.registry[v]["metrics"].get(metric, 0))
        return best_version

    def list_models(self, model_name: Optional[str] = None) -> List[Dict]:
        """List all models or models with specific name"""
        if model_name:
            return [self.registry[v] for v in self.registry.keys()
                   if self.registry[v]["model_name"] == model_name]
        return list(self.registry.values())

    def compare_models(self, model_name: str, metric: str = "f1_score") -> pd.DataFrame:
        """Compare all versions of a model"""
        model_versions = [v for v in self.registry.keys()
                         if self.registry[v]["model_name"] == model_name]

        if not model_versions:
            return pd.DataFrame()

        comparison_data = []
        for version in model_versions:
            info = self.registry[version]
            comparison_data.append({
                "version": version,
                "created_at": info["created_at"],
                "metric": info["metrics"].get(metric, 0),
                "accuracy": info["metrics"].get("accuracy", 0),
                "precision": info["metrics"].get("precision", 0),
                "recall": info["metrics"].get("recall", 0)
            })

        return pd.DataFrame(comparison_data).sort_values("metric", ascending=False)

class ABTestingFramework:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.experiments = {}

    def create_experiment(self, name: str, model_versions: List[str],
                         control_version: str) -> str:
        """Create A/B testing experiment"""
        experiment_id = f"exp_{len(self.experiments) + 1}_{hashlib.sha256(name.encode()).hexdigest()[:8]}"

        self.experiments[experiment_id] = {
            "name": name,
            "experiment_id": experiment_id,
            "model_versions": model_versions,
            "control_version": control_version,
            "created_at": datetime.now().isoformat(),
            "results": {},
            "status": "active"
        }

        logger.info(f"Created A/B experiment: {experiment_id}")
        return experiment_id

    def record_prediction(self, experiment_id: str, model_version: str,
                         prediction: Any, actual: Any = None):
        """Record prediction result for A/B testing"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        if model_version not in self.experiments[experiment_id]["model_versions"]:
            raise ValueError(f"Model version {model_version} not in experiment")

        if model_version not in self.experiments[experiment_id]["results"]:
            self.experiments[experiment_id]["results"][model_version] = {
                "predictions": [],
                "actuals": [],
                "accuracy": 0,
                "count": 0
            }

        self.experiments[experiment_id]["results"][model_version]["predictions"].append(prediction)
        if actual is not None:
            self.experiments[experiment_id]["results"][model_version]["actuals"].append(actual)

        # Update metrics
        results = self.experiments[experiment_id]["results"][model_version]
        results["count"] = len(results["predictions"])

        if actual is not None and len(results["actuals"]) > 0:
            correct = sum(1 for p, a in zip(results["predictions"], results["actuals"])
                         if p == a)
            results["accuracy"] = correct / len(results["actuals"])

    def get_experiment_results(self, experiment_id: str) -> Dict:
        """Get results of A/B testing experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        return self.experiments[experiment_id]

    def conclude_experiment(self, experiment_id: str, winner_version: str):
        """Conclude experiment and mark winner"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        self.experiments[experiment_id]["status"] = "concluded"
        self.experiments[experiment_id]["winner"] = winner_version
        self.experiments[experiment_id]["concluded_at"] = datetime.now().isoformat()

        logger.info(f"Experiment {experiment_id} concluded. Winner: {winner_version}")

def run_model_versioning_demo():
    """Demonstrate model versioning and A/B testing"""
    logger.info("🚀 Starting Model Versioning Demo")

    # Initialize registry and A/B testing
    registry = ModelRegistry()
    ab_testing = ABTestingFramework(registry)

    # Dummy training data with correlation for higher accuracy
    np.random.seed(42)  # For reproducibility
    X_train = np.random.rand(200, 5)
    y_train = ((X_train[:, 0] + X_train[:, 1] + X_train[:, 2]) * 1.25).astype(int).clip(0, 4)
    X_test = np.random.rand(100, 5)
    y_test = ((X_test[:, 0] + X_test[:, 1] + X_test[:, 2]) * 1.25).astype(int).clip(0, 4)

    # Register multiple model versions with different parameters
    versions = []
    for i in range(3):
        model_params = {
            "n_estimators": 50 + i * 50,  # 50, 100, 150
            "max_depth": 5 + i * 5,       # 5, 10, 15
            "random_state": 42
        }

        # Train the model
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)

        # Calculate actual metrics on test set
        predictions = model.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        precision = np.mean([1 if p == a and p != 0 else 0 for p, a in zip(predictions, y_test)]) / np.mean(predictions != 0) if np.any(predictions != 0) else 0
        recall = np.mean([1 if p == a and a != 0 else 0 for p, a in zip(predictions, y_test)]) / np.mean(y_test != 0) if np.any(y_test != 0) else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

        version = registry.register_model(
            model_name="solar_flare_rf",
            model=model,
            model_params=model_params,
            metrics=metrics,
            model_type="sklearn"
        )
        versions.append(version)
        logger.info(f"Registered model version: {version} with accuracy {accuracy:.3f}")

    # Create A/B testing experiment
    experiment_id = ab_testing.create_experiment(
        name="solar_flare_prediction_test",
        model_versions=versions,
        control_version=versions[0]
    )

    # Use actual model predictions for A/B testing
    for version in versions:
        model = registry.get_model(version)
        predictions = model.predict(X_test)
        for pred, actual in zip(predictions, y_test):
            ab_testing.record_prediction(experiment_id, version, int(pred), int(actual))

    # Get experiment results
    results = ab_testing.get_experiment_results(experiment_id)
    logger.info("A/B Testing Results:")
    for version, metrics in results["results"].items():
        logger.info(f"  {version}: Accuracy = {metrics['accuracy']:.3f}")

    # Conclude experiment
    best_version = max(results["results"].items(),
                      key=lambda x: x[1]["accuracy"])[0]
    ab_testing.conclude_experiment(experiment_id, best_version)

    logger.info("✅ Model versioning demo complete!")

if __name__ == "__main__":
    run_model_versioning_demo()
