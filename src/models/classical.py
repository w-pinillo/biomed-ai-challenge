import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import numpy as np
import os

# Using iterative_train_test_split for multi-label stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# Define paths
PREPROCESSED_PATH = "data/preprocessed_articles.csv"
MODEL_OUTPUT_DIR = "models"
MODEL_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, "classical_logreg.joblib")

# Define labels and features
LABELS = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']
TEXT_FEATURE = 'full_text_cleaned'
DOMAIN_FEATURES = [
    'cardiovascular_keywords_count',
    'neurological_keywords_count',
    'hepatorenal_keywords_count',
    'oncological_keywords_count'
]

def main():
    """
    Main function to train and evaluate the classical model.
    """
    print("Starting classical model training...")

    # Load data
    df = pd.read_csv(PREPROCESSED_PATH)
    print(f"Loaded {len(df)} records from {PREPROCESSED_PATH}")

    # Define X and y
    X = df[[TEXT_FEATURE] + DOMAIN_FEATURES]
    y = df[LABELS].values

    # Multi-label stratified split
    # n_splits=1 because we want a single train/test split
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index, test_index = next(msss.split(X, y))

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Create a column transformer to apply different transformations to different columns
    # We use TfidfVectorizer for the text and pass through the numeric domain features
    preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), TEXT_FEATURE),
            ('domain', 'passthrough', DOMAIN_FEATURES)
        ])

    # Create the full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=42)))
    ])

    # Train the model
    print("\nTraining the Logistic Regression model...")
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    print("\nEvaluating the model on the test set...")
    y_pred = pipeline.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=LABELS, zero_division=0)
    print("\n--- Classification Report ---")
    print(report)

    # Save the trained pipeline
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    print(f"\nSaving the trained model to {MODEL_OUTPUT_PATH}...")
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
