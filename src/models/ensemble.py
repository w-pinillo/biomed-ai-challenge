import pandas as pd
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import yaml

def load_config(config_path='config.yml'):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    """
    Main function to create and evaluate an ensemble of the classical and transformer models.
    """
    config = load_config()

    # Load data and embeddings
    df = pd.read_csv(config['preprocessed_data'])
    embeddings = np.load('data/biobert_embeddings.npy')
    
    # Define features and labels
    LABELS = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']
    DOMAIN_FEATURES = [
        'cardiovascular_keywords_count',
        'neurological_keywords_count',
        'hepatorenal_keywords_count',
        'oncological_keywords_count'
    ]
    X_domain = df[DOMAIN_FEATURES].values
    X = np.concatenate([embeddings, X_domain], axis=1)
    y = df[LABELS].values

    # Split data to get the same test set as in classical.py
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_index = next(msss.split(X, y))
    X_test = X[test_index]
    y_test = y[test_index]
    texts_test = df['full_text_cleaned'].iloc[test_index].tolist()

    # Load trained classical model
    classical_model = joblib.load('models/classical_logreg_bilstm.joblib')
    
    # Get predictions from classical model
    classical_pred_proba = classical_model.predict_proba(X_test)

    # Get predictions from pre-trained BioBERT model
    tokenizer = AutoTokenizer.from_pretrained(config['model_checkpoint'])
    model = AutoModelForSequenceClassification.from_pretrained(config['model_checkpoint'], num_labels=len(LABELS), problem_type="multi_label_classification")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    biobert_pred_proba = []
    batch_size = config['batch_size']
    for i in range(0, len(texts_test), batch_size):
        batch_texts = texts_test[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=config['max_length'])
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probas = torch.sigmoid(logits).cpu().numpy()
        biobert_pred_proba.extend(probas)
    
    biobert_pred_proba = np.array(biobert_pred_proba)

    # Find optimal weights
    best_f1 = 0
    best_weights = (0, 0)
    for i in np.arange(0, 1.1, 0.1):
        weight_classical = i
        weight_biobert = 1 - i
        ensemble_pred_proba = (weight_classical * classical_pred_proba) + (weight_biobert * biobert_pred_proba)
        ensemble_preds = (ensemble_pred_proba > 0.5).astype(int)
        f1 = f1_score(y_test, ensemble_preds, average='weighted')
        print(f"Weights: (classical: {weight_classical:.1f}, biobert: {weight_biobert:.1f}), F1-score: {f1:.4f})")
        if f1 > best_f1:
            best_f1 = f1
            best_weights = (weight_classical, weight_biobert)

    # Evaluate ensemble with best weights
    print(f"\nBest weights: (classical: {best_weights[0]:.1f}, biobert: {best_weights[1]:.1f}) with F1-score: {best_f1:.4f})")
    ensemble_pred_proba = (best_weights[0] * classical_pred_proba) + (best_weights[1] * biobert_pred_proba)
    ensemble_preds = (ensemble_pred_proba > 0.5).astype(int)
    
    print("\n--- Ensemble Classification Report with Best Weights ---")
    report = classification_report(y_test, ensemble_preds, target_names=LABELS, zero_division=0)
    print(report)

if __name__ == "__main__":
    main()
