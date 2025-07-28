import pandas as pd
import numpy as np
import argparse
import os
import json
import warnings
from datetime import datetime
from typing import Dict, Tuple, Any

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import metrics
from sklearn.inspection import permutation_importance

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Model persistence and experiment tracking
import joblib
from dvclive import Live

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class MLTrainingPipeline:
    """A comprehensive, reusable ML training pipeline."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.label_encoder = LabelEncoder()
        
        # Create output directories for this experiment run
        self.run_dir = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.plots_dir = os.path.join("plots", self.run_dir)
        self.reports_dir = os.path.join("reports", self.run_dir)
        self.models_dir = "models" # Models can be overwritten
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def load_and_preprocess_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads and preprocesses the dataset."""
        print("\n--- 1. Loading and Preprocessing Data ---")
        data = pd.read_csv(data_path)
        print(f"Dataset shape: {data.shape}, Missing values: {data.isnull().sum().sum()}")
        
        train_df, val_df = train_test_split(
            data, 
            test_size=self.config['test_size'], 
            stratify=data['species'], 
            random_state=self.config['random_state']
        )
        print(f"Train set: {len(train_df)} samples | Validation set: {len(val_df)} samples")
        return train_df, val_df
    
    def poison_data(self, df: pd.DataFrame, poison_rate: float) -> pd.DataFrame:
        """Applies mislabeling data poisoning to the dataframe."""
        if poison_rate == 0:
            print("\n--- 2. Data Poisoning ---\nNo data poisoning applied.")
            return df
        
        print(f"\n--- 2. Data Poisoning ---")
        n_poison = int(len(df) * poison_rate)
        print(f"Applying {poison_rate:.0%} mislabeling to {n_poison} training samples.")
        
        classes = df['species'].unique()
        poison_indices = np.random.choice(df.index, n_poison, replace=False)
        
        for i in poison_indices:
            original_class = df.loc[i, 'species']
            new_class = np.random.choice([c for c in classes if c != original_class])
            df.loc[i, 'species'] = new_class
            
        return df
    
    def prepare_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple:
        """Prepares features and labels, including scaling."""
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        X_train = train_df[feature_cols].values
        X_val = val_df[feature_cols].values
        
        y_train = self.label_encoder.fit_transform(train_df['species'])
        y_val = self.label_encoder.transform(val_df['species'])
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        joblib.dump(scaler, os.path.join(self.models_dir, 'scaler.pkl'))
        
        return X_train_scaled, X_val_scaled, y_train, y_val
        
    def train_and_evaluate_models(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Trains, cross-validates, and evaluates multiple models."""
        print("\n--- 3. Training and Evaluation ---")
        models_config = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=self.config['random_state']),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=self.config['random_state']),
            'SVM': SVC(probability=True, random_state=self.config['random_state'])
        }
        
        evaluation_results = {}
        cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config['random_state'])
        
        for name, model in models_config.items():
            print(f"\nProcessing Model: {name}")
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='accuracy')
            print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Final training and evaluation
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            val_proba = model.predict_proba(X_val)
            
            # Store metrics
            evaluation_results[name] = {
                'model': model,
                'val_accuracy': metrics.accuracy_score(y_val, val_pred),
                'val_loss': metrics.log_loss(y_val, val_proba),
                'val_f1': metrics.f1_score(y_val, val_pred, average='macro'),
                'confusion_matrix': metrics.confusion_matrix(y_val, val_pred)
            }
            print(f"  Validation Accuracy: {evaluation_results[name]['val_accuracy']:.4f}")
            joblib.dump(model, os.path.join(self.models_dir, f'{name}.pkl'))

        return evaluation_results

    def create_visualizations(self, evaluation_results: Dict):
        """Creates and saves visualizations for the experiment run."""
        print("\n--- 4. Creating Visualizations ---")
        sns.set_theme(style="whitegrid")
        
        # Confusion Matrices Plot
        n_models = len(evaluation_results)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        fig.suptitle(f"Confusion Matrices (Poison Rate: {self.config['poison_rate']:.0%})", fontsize=16)
        
        for idx, (name, results) in enumerate(evaluation_results.items()):
            ax = axes[idx] if n_models > 1 else axes
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                        xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_, ax=ax)
            ax.set_title(name)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.plots_dir, 'confusion_matrices.png'), dpi=300)
        plt.close()
        print(f"Saved confusion matrices plot to {self.plots_dir}")

    def run(self):
        """Executes the entire pipeline."""
        with Live(save_dvc_exp=True) as live:
            # Log parameters
            live.log_params(self.config)
            
            # Run pipeline steps
            train_df, val_df = self.load_and_preprocess_data(self.config['data_path'])
            poisoned_train_df = self.poison_data(train_df, self.config['poison_rate'])
            X_train, X_val, y_train, y_val = self.prepare_features(poisoned_train_df, val_df)
            eval_results = self.train_and_evaluate_models(X_train, y_train, X_val, y_val)
            self.create_visualizations(eval_results)
            
            # Log metrics for the best model (by accuracy)
            best_model_name = max(eval_results, key=lambda k: eval_results[k]['val_accuracy'])
            live.log_metric("best_model_accuracy", eval_results[best_model_name]['val_accuracy'])
            live.log_metric("best_model_loss", eval_results[best_model_name]['val_loss'])
            live.log_metric("best_model_f1", eval_results[best_model_name]['val_f1'])
            
            print("\n--- 5. Experiment Complete ---")
            print(f"Best performing model: {best_model_name}")

def main():
    parser = argparse.ArgumentParser(description="Advanced ML Training Pipeline")
    parser.add_argument("--poison-rate", type=float, default=0.0)
    args = parser.parse_args()
    
    config = {
        'data_path': 'data/iris.csv',
        'test_size': 0.3,
        'random_state': 42,
        'poison_rate': args.poison_rate
    }
    
    pipeline = MLTrainingPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()
