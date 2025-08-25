#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting end-to-end pipeline..."

# Step 1: Preprocessing
echo "Running data preprocessing (src/preprocess.py)..."
# This script loads raw data, cleans it, creates domain features, and saves preprocessed_articles.csv
venv/bin/python src/preprocess.py

# Step 2: Classical Model Training and Hyperparameter Tuning
echo "Running classical model training and hyperparameter tuning (src/models/classical.py)..."
# This script loads preprocessed data and BioBERT embeddings, performs Optuna tuning for Logistic Regression,
# trains the best model, and saves it.
venv/bin/python -m src.models.classical

# Step 3: Ensemble Model Evaluation and Error Analysis
echo "Running ensemble model evaluation and error analysis (src/models/ensemble.py)..."
# This script loads the tuned classical model, gets predictions from base BioBERT,
# finds optimal ensemble weights, evaluates, and performs error analysis.
venv/bin/python -m src.models.ensemble

echo "End-to-end pipeline completed successfully."
