import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
warnings.filterwarnings('ignore')

class SolarFlarePredictor:
    def __init__(self, data_path=None):
        self.project_root = Path(__file__).resolve().parent.parent
        self.data_path = data_path or self.project_root / "data" / "historical_goes_2010_2015_parsed.csv"
        self.model_dir = self.project_root / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize results storage
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0

    def load_and_preprocess_data(self):
        """Load and preprocess the solar flare dataset"""
        print("Loading data from:", self.data_path)
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

        print(f"Dataset shape: {X.shape}")
        print(f"Class distribution:\n{y.value_counts().sort_index()}")

        return X, y, feature_columns

    def handle_class_imbalance(self, X, y):
        """Handle class imbalance using SMOTE"""
        # Check class distribution
        class_counts = y.value_counts().sort_index()
        print(f"Class distribution before SMOTE:\n{class_counts}")

        # Remove classes with very few samples (less than k_neighbors + 1)
        min_samples_needed = 6  # Default k_neighbors for SMOTE
        valid_classes = class_counts[class_counts >= min_samples_needed].index
        print(f"Valid classes (with >= {min_samples_needed} samples): {valid_classes.tolist()}")

        # Filter data to keep only valid classes
        mask = y.isin(valid_classes)
        X_filtered = X[mask]
        y_filtered = y[mask]

        print(f"Filtered dataset shape: {X_filtered.shape}")
        print(f"Filtered class distribution:\n{y_filtered.value_counts().sort_index()}")

        # Apply SMOTE only if we have enough samples
        if len(y_filtered.unique()) > 1 and len(y_filtered) >= min_samples_needed:
            smote = SMOTE(random_state=42, k_neighbors=min(5, len(y_filtered.unique())-1))
            X_resampled, y_resampled = smote.fit_resample(X_filtered, y_filtered)
            print(f"After SMOTE - New shape: {X_resampled.shape}")
            print(f"New class distribution:\n{y_resampled.value_counts().sort_index()}")
            return X_resampled, y_resampled
        else:
            print("Not enough samples for SMOTE, using original data")
            return X_filtered, y_filtered

    def get_models_and_params(self):
        """Define models and their hyperparameters for tuning"""
        models_and_params = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'LightGBM': {
                'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'CatBoost': {
                'model': cb.CatBoostClassifier(random_state=42, verbose=False),
                'params': {
                    'iterations': [100, 200, 300],
                    'depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'Neural Network': {
                'model': self._create_neural_network(),
                'params': {
                    'epochs': [50, 100],
                    'batch_size': [16, 32],
                    'learning_rate': [0.001, 0.01]
                }
            }
        }
        return models_and_params

    def _create_neural_network(self):
        """Create a neural network model for solar flare prediction"""
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
            Dense(5, activation='softmax')  # 5 classes for flare prediction
        ])
        return model

    def train_and_evaluate_model(self, name, model, params, X_train, X_test, y_train, y_test):
        """Train and evaluate a single model with hyperparameter tuning"""
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print('='*50)

        # Handle neural networks differently
        if name == 'Neural Network':
            return self._train_neural_network(model, params, X_train, X_test, y_train, y_test)

        # Perform grid search for traditional ML models
        grid_search = GridSearchCV(
            model, params, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)

        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"Best parameters: {best_params}")

        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)

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
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1_weighted')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # Store results
        self.results[name] = {
            'model': best_model,
            'best_params': best_params,
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

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        if roc_auc:
            print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"CV Score: {cv_mean:.4f} (+/- {cv_std*2:.4f})")

        return f1  # Return F1 score for comparison

    def _train_neural_network(self, model_template, params, X_train, X_test, y_train, y_test):
        """Train neural network with hyperparameter tuning"""
        from itertools import product

        # Get all combinations of hyperparameters
        param_combinations = list(product(*params.values()))
        param_names = list(params.keys())

        best_score = 0
        best_model = None
        best_params = None

        print(f"Testing {len(param_combinations)} hyperparameter combinations...")

        for i, param_values in enumerate(param_combinations):
            print(f"\n--- Combination {i+1}/{len(param_combinations)} ---")

            # Extract parameters
            param_dict = dict(zip(param_names, param_values))
            epochs = param_dict['epochs']
            batch_size = param_dict['batch_size']
            learning_rate = param_dict['learning_rate']

            print(f"Parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

            # Create fresh model
            model = tf.keras.models.clone_model(model_template)
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # Set up callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            )

            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=0
            )

            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )

            # Evaluate on test set
            y_pred_proba = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # ROC-AUC for multiclass
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            except:
                roc_auc = None

            print(f"Results: Accuracy={accuracy:.4f}, F1={f1:.4f}")
            if roc_auc:
                print(f"ROC-AUC: {roc_auc:.4f}")

            # Track best model
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_params = param_dict

        print(f"\nBest Neural Network parameters: {best_params}")
        print(f"Best Neural Network F1-Score: {best_score:.4f}")

        # Store results
        self.results['Neural Network'] = {
            'model': best_model,
            'best_params': best_params,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': best_score,
            'roc_auc': roc_auc,
            'cv_mean': None,  # Not applicable for neural networks
            'cv_std': None,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        return best_score

    def train_all_models(self):
        """Train and evaluate all models"""
        # Load and preprocess data
        X, y, feature_columns = self.load_and_preprocess_data()

        # Handle class imbalance
        X_resampled, y_resampled = self.handle_class_imbalance(X, y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Get models and parameters
        models_and_params = self.get_models_and_params()

        # Train all models
        for name, model_info in models_and_params.items():
            try:
                score = self.train_and_evaluate_model(
                    name, model_info['model'], model_info['params'],
                    X_train_scaled, X_test_scaled, y_train, y_test
                )

                # Track best model
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = self.results[name]['model']
                    self.best_model_name = name

            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue

    def print_comparison_table(self):
        """Print a comparison table of all models"""
        print(f"\n{'='*80}")
        print("MODEL COMPARISON RESULTS")
        print('='*80)
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10} {'CV Score':<10}")
        print('-' * 80)

        for name, metrics in self.results.items():
            roc_auc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A"
            cv_str = f"{metrics['cv_mean']:.4f}"
            print(f"{name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} {roc_auc_str:<10} {cv_str:<10}")

        print('=' * 80)
        print(f"🏆 Best Model: {self.best_model_name} (F1-Score: {self.best_score:.4f})")

    def save_best_model(self):
        """Save the best performing model"""
        if self.best_model:
            # Handle neural network models differently
            if self.best_model_name == 'Neural Network':
                # Save neural network as HDF5
                model_path = self.model_dir / "best_solar_flare_model.h5"
                self.best_model.save(model_path)
                print(f"\n💾 Best neural network model saved to: {model_path}")

                # Save model metadata
                metadata = {
                    'model_name': self.best_model_name,
                    'best_score': self.best_score,
                    'training_date': datetime.now().isoformat(),
                    'feature_columns': ['flux', 'month', 'day', 'hour', 'day_of_year'],
                    'model_type': 'neural_network'
                }
            else:
                # Save traditional ML models as pickle
                model_path = self.model_dir / "best_solar_flare_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(self.best_model, f)

                # Also save as joblib for compatibility
                joblib_path = self.model_dir / "best_solar_flare_model.joblib"
                joblib.dump(self.best_model, joblib_path)

                print(f"\n💾 Best model saved to: {model_path}")
                print(f"💾 Also saved as: {joblib_path}")

                # Save model metadata
                metadata = {
                    'model_name': self.best_model_name,
                    'best_score': self.best_score,
                    'training_date': datetime.now().isoformat(),
                    'feature_columns': ['flux', 'month', 'day', 'hour', 'day_of_year'],
                    'model_type': 'traditional_ml'
                }

            metadata_path = self.model_dir / "model_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

    def plot_comparison(self):
        """Create comparison plots"""
        # Extract metrics for plotting
        model_names = list(self.results.keys())
        f1_scores = [self.results[name]['f1_score'] for name in model_names]
        accuracies = [self.results[name]['accuracy'] for name in model_names]

        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # F1-Score comparison
        ax1.bar(model_names, f1_scores, color='skyblue')
        ax1.set_title('F1-Score Comparison')
        ax1.set_ylabel('F1-Score')
        ax1.tick_params(axis='x', rotation=45)

        # Accuracy comparison
        ax2.bar(model_names, accuracies, color='lightgreen')
        ax2.set_title('Accuracy Comparison')
        ax2.set_ylabel('Accuracy')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.model_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Comparison plot saved to: {self.model_dir / 'model_comparison.png'}")

    def run_complete_analysis(self):
        """Run the complete multi-algorithm analysis"""
        print("🚀 Starting Multi-Algorithm Solar Flare Prediction Analysis")
        print("=" * 60)

        # Train all models
        self.train_all_models()

        # Print comparison table
        self.print_comparison_table()

        # Save best model
        self.save_best_model()

        # Create comparison plots
        self.plot_comparison()

        print("\n✅ Analysis complete!")
        print(f"📈 Best performing model: {self.best_model_name}")
        print(f"🎯 Best F1-Score: {self.best_score:.4f}")

        return self.best_model_name, self.best_score

if __name__ == "__main__":
    # Initialize predictor
    predictor = SolarFlarePredictor()

    # Run complete analysis
    best_model, best_score = predictor.run_complete_analysis()
