import pandas as pd
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.metrics import classification_report, f1_score
import yaml
import argparse
import re
import os

# Import error analysis function
from src.eval import analyze_errors

# --- Configuration Loading ---
def load_config(config_path='config.yml'):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# --- Preprocessing Functions (replicated from src/preprocess.py) ---
KEYWORD_SETS = {
    'cardiovascular_keywords': ['heart', 'cardiac', 'blood', 'artery', 'vein', 'vascular', 'hypertension', 'myocardial', 'coronary'],
    'neurological_keywords': ['brain', 'neuro', 'nerve', 'spinal', 'stroke', 'alzheimer', 'parkinson', 'epilepsy', 'neuron'],
    'hepatorenal_keywords': ['liver', 'hepatic', 'kidney', 'renal', 'hepatitis', 'nephropathy', 'dialysis', 'cirrhosis'],
    'oncological_keywords': ['cancer', 'tumor', 'oncology', 'chemotherapy', 'carcinoma', 'sarcoma', 'lymphoma', 'metastasis']
}
LABELS = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']

def clean_text(text):
    """
    Cleans a given text string by lowercasing and removing excess whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_domain_features(text, keyword_sets):
    """
    Creates domain-specific features by counting keywords in the text.
    Returns a dictionary of feature counts.
    """
    if not isinstance(text, str):
        return {f'{key}_count': 0 for key in keyword_sets.keys()}

    features = {}
    for key, keywords in keyword_sets.items():
        count = sum(1 for keyword in keywords if re.search(r'\b' + keyword + r'\b', text))
        features[f'{key}_count'] = count
    return features

# --- Feature Extraction (BioBERT Embeddings) ---
def get_biobert_embeddings(texts, tokenizer, model, device, max_length, batch_size):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.append(embeddings)
        
    return np.vstack(all_embeddings)

# --- Main Evaluation Function ---
def evaluate_solution(input_csv_path, separator):
    config = load_config()

    print(f"Loading data from {input_csv_path} with separator '{separator}'...")
    df = pd.read_csv(input_csv_path, sep=separator)
    print(f"Loaded {len(df)} records.")

    # --- Preprocessing ---
    print("Applying preprocessing steps...")
    df['title_cleaned'] = df['title'].apply(clean_text)
    df['abstract_cleaned'] = df['abstract'].apply(clean_text)
    df['full_text_cleaned'] = df['title_cleaned'] + ' ' + df['abstract_cleaned']

    domain_features_df = df['full_text_cleaned'].apply(lambda x: create_domain_features(x, KEYWORD_SETS)).apply(pd.Series)
    df = pd.concat([df, domain_features_df], axis=1)

    # --- Load Models ---
    print("Loading models...")
    # Load tuned classical model
    classical_model_path = os.path.join("models", "classical_logistic_regression_tuned.joblib")
    classical_model = joblib.load(classical_model_path)

    # Load BioBERT tokenizer and model for embeddings and direct predictions
    tokenizer = AutoTokenizer.from_pretrained(config['model_checkpoint'])
    biobert_embedding_model = AutoModel.from_pretrained(config['model_checkpoint'])
    biobert_classifier_model = AutoModelForSequenceClassification.from_pretrained(config['model_checkpoint'], num_labels=len(LABELS), problem_type="multi_label_classification")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    biobert_embedding_model.to(device)
    biobert_classifier_model.to(device)

    # --- Feature Extraction (Embeddings) ---
    print("Generating BioBERT embeddings...")
    texts_to_process = df['full_text_cleaned'].astype(str).tolist()
    embeddings = get_biobert_embeddings(texts_to_process, tokenizer, biobert_embedding_model, device, config['max_length'], config['batch_size'])

    # --- Prepare Features for Classical Model ---
    X_domain = df[[f'{key}_count' for key in KEYWORD_SETS.keys()]].values
    X_classical = np.concatenate([embeddings, X_domain], axis=1)

    # --- Get Predictions ---
    print("Generating predictions...")
    # Classical model predictions
    classical_pred_proba = classical_model.predict_proba(X_classical)

    # BioBERT base model predictions
    biobert_pred_proba = []
    for i in range(0, len(texts_to_process), config['batch_size']):
        batch_texts = texts_to_process[i:i+config['batch_size']]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=config['max_length'])
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = biobert_classifier_model(**inputs)
        logits = outputs.logits
        probas = torch.sigmoid(logits).cpu().numpy()
        biobert_pred_proba.extend(probas)
    biobert_pred_proba = np.array(biobert_pred_proba)

    # Ensemble predictions (using best weights from training phase)
    # NOTE: These weights are hardcoded from the last ensemble run (classical: 0.6, biobert: 0.4)
    # For a more robust solution, these could be loaded from a config or a saved ensemble model.
    best_classical_weight = 0.6
    best_biobert_weight = 0.4
    ensemble_pred_proba = (best_classical_weight * classical_pred_proba) + (best_biobert_weight * biobert_pred_proba)
    ensemble_preds = (ensemble_pred_proba > 0.5).astype(int)

    # Add predicted labels to DataFrame
    df['group_predicted'] = ['|'.join([LABELS[j] for j, pred in enumerate(row) if pred == 1]) for row in ensemble_preds]
    df['group_predicted'] = df['group_predicted'].replace({'': 'None'}) # Handle cases with no predicted labels

    # --- Evaluation (if 'group' column exists) ---
    if 'group' in df.columns:
        print("\nPerforming evaluation...")
        # Convert true labels to one-hot encoding for evaluation
        y_true_one_hot = np.zeros((len(df), len(LABELS)))
        for idx, row in df.iterrows():
            true_labels = [label.lower() for label in row['group'].split('|')] if pd.notna(row['group']) else []
            for j, label_name in enumerate(LABELS):
                if label_name.lower() in true_labels:
                    y_true_one_hot[idx, j] = 1
        
        # Classification Report
        print("\n--- Classification Report ---")
        report = classification_report(y_true_one_hot, ensemble_preds, target_names=LABELS, zero_division=0)
        print(report)

        # Weighted F1-score (main metric)
        weighted_f1 = f1_score(y_true_one_hot, ensemble_preds, average='weighted', zero_division=0)
        print(f"\nWeighted F1-score: {weighted_f1:.4f}")

        # Confusion Matrix and Error Analysis
        analyze_errors(y_true_one_hot, ensemble_preds, df['full_text_cleaned'].tolist(), LABELS)
    else:
        print("\n'group' column not found in input CSV. Skipping evaluation.")

    print("\n--- Predictions with 'group_predicted' column ---")
    print(df[['title', 'abstract', 'group', 'group_predicted']].head())

    return df

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate medical literature classification solution.")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the input CSV file (e.g., data/medical_articles.csv).")
    parser.add_argument('--separator', type=str, default=';', help="CSV file separator (e.g., ';', ',', '\t'). Defaults to ';'.")
    args = parser.parse_args()

    # Ensure the script is run from the project root for correct path resolution
    if not os.path.exists('config.yml'):
        print("Error: Please run this script from the project's root directory.")
        exit(1)

    evaluate_solution(args.input_csv, args.separator)