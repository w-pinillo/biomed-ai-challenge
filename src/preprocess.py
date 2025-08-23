import pandas as pd
import re
import os

# Define paths
DATA_PATH = "data/medical_articles.csv"
PREPROCESSED_PATH = "data/preprocessed_articles.csv"

# Define keywords for domain-specific features
# These are examples and can be expanded
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
        # Use word boundaries (\b) to match whole words only
        count = sum(1 for keyword in keywords if re.search(r'\b' + keyword + r'\b', text))
        features[f'{key}_count'] = count
    return features

def main():
    """
    Main function to run the preprocessing pipeline.
    """
    print("Starting preprocessing...")

    # Load data
    df = pd.read_csv(DATA_PATH, sep=';')
    print(f"Loaded {len(df)} records from {DATA_PATH}")

    # Clean text columns
    print("Cleaning 'title' and 'abstract' columns...")
    df['title_cleaned'] = df['title'].apply(clean_text)
    df['abstract_cleaned'] = df['abstract'].apply(clean_text)
    df['full_text_cleaned'] = df['title_cleaned'] + ' ' + df['abstract_cleaned']

    # Create domain-specific features
    print("Creating domain-specific features from full text...")
    domain_features_df = df['full_text_cleaned'].apply(lambda x: create_domain_features(x, KEYWORD_SETS)).apply(pd.Series)
    df = pd.concat([df, domain_features_df], axis=1)

    # Re-create the one-hot encoded labels as they are needed for training
    df['labels'] = df['group'].fillna('').str.strip().str.split('|')
    for label in LABELS:
        df[label] = df['labels'].apply(lambda x: 1 if label.lower() in [l.lower() for l in x] else 0)

    # Select columns to save
    feature_columns = [f'{key}_count' for key in KEYWORD_SETS.keys()]
    columns_to_save = ['title', 'abstract', 'group', 'full_text_cleaned'] + feature_columns + LABELS
    
    df_to_save = df[columns_to_save]

    # Ensure data directory exists for preprocessed file
    os.makedirs(os.path.dirname(PREPROCESSED_PATH), exist_ok=True)
    print(f"Saving preprocessed data to {PREPROCESSED_PATH}...")
    df_to_save.to_csv(PREPROCESSED_PATH, index=False)
    
    print("Preprocessing finished successfully.")
    print(f"Preprocessed data has {len(df_to_save.columns)} columns.")
    print("\nExample of preprocessed data:")
    print(df_to_save.head())

if __name__ == "__main__":
    main()
