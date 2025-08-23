import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import yaml
from tqdm import tqdm

def load_config(config_path='config.yml'):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    """
    Main function to extract features using a pre-trained transformer model.
    """
    config = load_config()
    
    print("Loading preprocessed data...")
    df = pd.read_csv(config['preprocessed_data'])
    texts = df['full_text_cleaned'].astype(str).tolist()

    print(f"Loading tokenizer and model from {config['model_checkpoint']}...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_checkpoint'])
    model = AutoModel.from_pretrained(config['model_checkpoint'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Extracting embeddings in batches using {device}...")
    batch_size = config['batch_size']
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting Embeddings"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=config['max_length'])
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Using mean of last hidden state as sentence embedding
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.append(embeddings)
        
    embeddings_array = np.vstack(all_embeddings)
    
    # Save embeddings
    embeddings_output_path = 'data/biobert_embeddings.npy'
    print(f"Saving embeddings to {embeddings_output_path}...")
    np.save(embeddings_output_path, embeddings_array)
    print("Embeddings saved successfully.")

if __name__ == "__main__":
    main()
