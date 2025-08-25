import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import os
import optuna # Added Optuna import
import argparse # Added argparse import

def load_model(model_path):
    print(f"Loading pre-trained model from {model_path}...")
    return joblib.load(model_path)

# Using iterative_train_test_split for multi-label stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress ConvergenceWarning for this script
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Define paths
PREPROCESSED_PATH = "data/preprocessed_articles.csv"
EMBEDDINGS_PATH = "data/biobert_embeddings.npy"
MODEL_OUTPUT_DIR = "models"

# Define labels and features
LABELS = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']
DOMAIN_FEATURES = [
    'cardiovascular_keywords_count',
    'neurological_keywords_count',
    'hepatorenal_keywords_count',
    'oncological_keywords_count'
]

# Objective function for Optuna
def objective(trial, X_train, y_train, X_val, y_val, model_type):
    if model_type == "Logistic Regression":
        # Hyperparameters for Logistic Regression
        c = trial.suggest_float('lr_c', 1e-3, 1e3, log=True)
        solver = trial.suggest_categorical('lr_solver', ['liblinear', 'saga']) # saga for larger datasets, liblinear for smaller
        
        # Set n_jobs based on solver
        n_jobs = -1 if solver != 'liblinear' else 1
        
        classifier = LogisticRegression(C=c, solver=solver, random_state=42, n_jobs=n_jobs)
        
    else:
        raise ValueError("Unknown model type")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', OneVsRestClassifier(classifier))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    
    # Return weighted F1-score for multi-label classification
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    return f1

def main():
    parser = argparse.ArgumentParser(description="Classical model training and hyperparameter tuning.")
    parser.add_argument('--skip-training', action='store_true', help="Skip training and load pre-trained model.")
    args = parser.parse_args()

    # Define model path (should be consistent with where it's saved)
    lr_model_path = os.path.join(MODEL_OUTPUT_DIR, "classical_logistic_regression_tuned.joblib")

    # Load data (common to both training and loading paths)
    df = pd.read_csv(PREPROCESSED_PATH)
    print(f"Loaded {len(df)} records from {PREPROCESSED_PATH}")

    # Load BioBERT embeddings
    print(f"Loading BioBERT embeddings from {EMBEDDINGS_PATH}...")
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"Loaded embeddings of shape: {embeddings.shape}")

    # Define X and y
    X_domain = df[DOMAIN_FEATURES].values
    X = np.concatenate([embeddings, X_domain], axis=1)
    y = df[LABELS].values

    # Multi-label stratified split for train/val/test
    msss_train_val_test = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_index, test_index = next(msss_train_val_test.split(X, y))

    X_train_val, X_test = X[train_val_index], X[test_index]
    y_train_val, y_test = y[train_val_index], y[test_index]

    if args.skip_training:
        if os.path.exists(lr_model_path):
            final_lr_pipeline = load_model(lr_model_path)
        else:
            print(f"Error: Pre-trained model not found at {lr_model_path}. Training must be performed or ensure model exists.")
            exit(1)
    else:
        print("Starting classical model hyperparameter tuning with Optuna...")
        # Second split: 80% train, 20% val from train_val set
        msss_train_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42) # 0.25 of 0.8 is 0.2
        train_index, val_index = next(msss_train_val.split(X_train_val, y_train_val))

        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]

        print(f"Train set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")

        # --- Hyperparameter Tuning for Logistic Regression ---
        print("\n--- Starting Optuna tuning for Logistic Regression ---")
        study_lr = optuna.create_study(direction="maximize")
        study_lr.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, "Logistic Regression"), n_trials=50) # Adjust n_trials as needed

        print("\nBest trial for Logistic Regression:")
        print(f"  Value: {study_lr.best_value:.4f}")
        print(f"  Params: {study_lr.best_params}")

        # Train final Logistic Regression model with best params on train+val set
        best_lr_params = study_lr.best_params

        # Set n_jobs based on the best solver
        final_solver = best_lr_params['lr_solver']
        final_n_jobs = -1 if final_solver != 'liblinear' else 1

        final_lr_classifier = LogisticRegression(C=best_lr_params['lr_c'], solver=final_solver, random_state=42, n_jobs=final_n_jobs)
        final_lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', OneVsRestClassifier(final_lr_classifier))
        ])

        print("\nTraining final Logistic Regression model with best parameters on train+val set...")
        final_lr_pipeline.fit(X_train_val, y_train_val)

        # Save the tuned Logistic Regression model
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        joblib.dump(final_lr_pipeline, lr_model_path)
        print(f"Tuned Logistic Regression model saved to {lr_model_path}")

    # Evaluation part (common to both training and loading)
    print("\nEvaluating final Logistic Regression model on the test set...")
    y_pred_lr = final_lr_pipeline.predict(X_test)
    report_lr = classification_report(y_test, y_pred_lr, target_names=LABELS, zero_division=0)
    print("\n--- Final Classification Report for Tuned Logistic Regression ---")
    print(report_lr)

if __name__ == "__main__":
    main()