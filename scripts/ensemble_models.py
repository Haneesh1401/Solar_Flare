import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import warnings
import logging
from typing import Dict, List, Any, Optional

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SolarFlareEnsemble:
    def __init__(self, data_path=None):
        self.project_root = Path(__file__).resolve().parent.parent
        self.data_path = data_path or self.project_root / "data" / "historical_goes_2010_2015_parsed.csv"
        self.model_dir = self.project_root / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize results storage
        self.results = {}
        self.base_models = {}
        self.ensemble_models = {}

    def load_and_preprocess_data(self):
        """Load and preprocess the solar flare dataset"""
        logger.info("Loading data from: %s", self.data_path)
        df = pd.read_csv(self.data_path)

        # Handle missing values
        df = df.dropna(subset=['flare_class', 'flux', 'start'])
        df['flux'] = pd.to_numeric(df['flux'], errors='coerce')
        df = df.dropna(subset=['flux'])

        # Convert datetime
        df['start'] = pd.to_datetime(df['start'], errors='coerce')
        df = df.dropna(subset=['start'])

        # Feature engineering
        df['month'] = df['start'].dt.month
        df['day'] = df['start'].dt.day
        df['hour'] = df['start'].dt.hour
        df['day_of_year'] = df['start'].dt.dayofyear

        # Create flare class mapping
        flare_class_map = {'NO FLARE': 0, 'B': 1, 'C': 2, 'M': 3, 'X': 4}

        def convert_flare_class(x):
            try:
                val = float(x)
                if val < 10:
                    return 0
                elif val < 20:
                    return 1
                elif val < 40:
                    return 2
                elif val < 60:
                    return 3
                else:
                    return 4
            except:
                return flare_class_map.get(str(x).upper(), 0)

        df['flare_class_num'] = df['flare_class'].apply(convert_flare_class)

        # Features and target
        feature_columns = ['flux', 'month', 'day', 'hour', 'day_of_year']
        X = df[feature_columns]
        y = df['flare_class_num']

        logger.info("Dataset shape: %s", X.shape)
        logger.info("Class distribution:\n%s", y.value_counts().sort_index())

        return X, y, feature_columns

    def create_base_models(self):
        """Create base models for ensemble"""
        base_models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=42
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200, max_depth=7, learning_rate=0.1, random_state=42
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200, max_depth=7, learning_rate=0.1, random_state=42, verbose=-1
            ),
            'CatBoost': cb.CatBoostClassifier(
                iterations=200, depth=7, learning_rate=0.1, random_state=42, verbose=False
            ),
            'SVM': SVC(
                C=1.0, kernel='rbf', probability=True, random_state=42
            ),
            'Neural Network': self._create_neural_network()
        }
        return base_models

    def _create_neural_network(self):
        """Create a neural network model for ensemble"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(5,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            Dense(5, activation='softmax')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_base_models(self, X_train, y_train):
        """Train all base models"""
        logger.info("Training base models...")

        base_models = self.create_base_models()

        for name, model in base_models.items():
            logger.info("Training %s...", name)

            if name == 'Neural Network':
                # Handle neural network training
                model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
            else:
                model.fit(X_train, y_train)

            self.base_models[name] = model
            logger.info("%s trained successfully", name)

    def create_ensemble_models(self):
        """Create different ensemble models"""
        base_estimators = [(name, model) for name, model in self.base_models.items()
                          if name != 'Neural Network']  # Exclude NN from voting for simplicity

        ensemble_models = {
            'Voting_Soft': VotingClassifier(
                estimators=base_estimators,
                voting='soft'
            ),
            'Voting_Hard': VotingClassifier(
                estimators=base_estimators,
                voting='hard'
            ),
            'Stacking_RF': StackingClassifier(
                estimators=base_estimators,
                final_estimator=RandomForestClassifier(n_estimators=100, random_state=42)
            ),
            'Stacking_LR': StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(random_state=42)
            )
        }

        return ensemble_models

    def train_ensemble_models(self, X_train, y_train):
        """Train all ensemble models"""
        logger.info("Training ensemble models...")

        ensemble_models = self.create_ensemble_models()

        for name, model in ensemble_models.items():
            logger.info("Training %s...", name)
            model.fit(X_train, y_train)
            self.ensemble_models[name] = model
            logger.info("%s trained successfully", name)

    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and store results"""
        logger.info("Evaluating all models...")

        all_models = {**self.base_models, **self.ensemble_models}

        for name, model in all_models.items():
            logger.info("Evaluating %s...", name)

            # Make predictions
            if name == 'Neural Network':
                y_pred_proba = model.predict(X_test, verbose=0)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # ROC-AUC (handle binary vs multiclass)
            try:
                if len(np.unique(y_test)) == 2:
                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            except:
                roc_auc = None

            # Cross-validation score
            try:
                if name != 'Neural Network':
                    cv_scores = cross_val_score(model, X_test, y_test, cv=3, scoring='f1_weighted')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                else:
                    cv_mean = cv_std = None
            except:
                cv_mean = cv_std = None

            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }

            logger.info("%s - F1: %.4f, Accuracy: %.4f", name, f1, accuracy)

    def print_comparison_table(self):
        """Print a comparison table of all models"""
        print(f"\n{'='*100}")
        print("ENSEMBLE MODEL COMPARISON RESULTS")
        print('='*100)
        print(f"{'Model'"<20"} {'Accuracy'"<10"} {'Precision'"<10"} {'Recall'"<10"} {'F1-Score'"<10"} {'ROC-AUC'"<10"} {'CV Score'"<10"}")
        print('-' * 100)

        for name, metrics in self.results.items():
            roc_auc_str = f"{metrics['roc_auc']".4f"}" if metrics['roc_auc'] else "N/A"
            cv_str = f"{metrics['cv_mean']".4f"}" if metrics['cv_mean'] else "N/A"
            print(f"{name"<20"} {metrics['accuracy']"<10.4f"} {metrics['precision']"<10.4f"} {metrics['recall']"<10.4f"} {metrics['f1_score']"<10.4f"} {roc_auc_str"<10"} {cv_str"<10"}")

        print('=' * 100)

        # Find best model
        best_model = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        print(f"🏆 Best Model: {best_model[0]} (F1-Score: {best_model[1]['f1_score']".4f"})")

    def save_best_ensemble_model(self):
        """Save the best performing ensemble model"""
        if not self.results:
            logger.warning("No results available. Run evaluation first.")
            return

        # Find best ensemble model (not base model)
        ensemble_results = {k: v for k, v in self.results.items() if k in self.ensemble_models}
        if not ensemble_results:
            logger.warning("No ensemble models found in results.")
            return

        best_ensemble = max(ensemble_results.items(), key=lambda x: x[1]['f1_score'])
        best_name, best_metrics = best_ensemble

        # Save the model
        model_path = self.model_dir / f"best_ensemble_{best_name.lower().replace(' ', '_')}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.ensemble_models[best_name], f)

        logger.info("Best ensemble model saved to: %s", model_path)

        # Save metadata
        metadata = {
            'model_name': best_name,
            'best_score': best_metrics['f1_score'],
            'training_date': datetime.now().isoformat(),
            'model_type': 'ensemble',
            'base_models': list(self.base_models.keys()),
            'metrics': best_metrics
        }

        metadata_path = self.model_dir / f"ensemble_metadata_{best_name.lower().replace(' ', '_')}.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info("Model metadata saved to: %s", metadata_path)

    def run_complete_ensemble_analysis(self):
        """Run the complete ensemble analysis"""
        logger.info("🚀 Starting Ensemble Solar Flare Prediction Analysis")
        print("=" * 70)

        # Load and preprocess data
        X, y, feature_columns = self.load_and_preprocess_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train base models
        self.train_base_models(X_train_scaled, y_train)

        # Train ensemble models
        self.train_ensemble_models(X_train_scaled, y_train)

        # Evaluate all models
        self.evaluate_models(X_test_scaled, y_test)

        # Print comparison table
        self.print_comparison_table()

        # Save best ensemble model
        self.save_best_ensemble_model()

        print("\n✅ Ensemble analysis complete!")

        # Return best model info
        best_ensemble = max(
            {k: v for k, v in self.results.items() if k in self.ensemble_models}.items(),
            key=lambda x: x[1]['f1_score']
        )

        return best_ensemble[0], best_ensemble[1]['f1_score']

if __name__ == "__main__":
    # Initialize ensemble predictor
    ensemble_predictor = SolarFlareEnsemble()

    # Run complete analysis
    best_model, best_score = ensemble_predictor.run_complete_ensemble_analysis()
