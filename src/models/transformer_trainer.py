import pandas as pd
import numpy as np
import torch
import yaml
import os
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# --- Configuration Loading ---
def load_config(config_path='config.yml'):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# --- Reproducibility ---
def set_seed(seed_value):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

# --- Dataset Class ---
class MedicalDataset(torch.utils.data.Dataset):
    """Custom PyTorch Dataset for our medical text data."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# --- Metrics Calculation ---
def compute_metrics(p):
    """Computes and returns a dictionary of metrics for evaluation."""
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # Apply sigmoid to logits and threshold
    sigmoid_preds = 1 / (1 + np.exp(-preds))
    binary_preds = (sigmoid_preds > 0.5).astype(int)
    labels = p.label_ids

    f1_micro = f1_score(y_true=labels, y_pred=binary_preds, average='micro')
    f1_macro = f1_score(y_true=labels, y_pred=binary_preds, average='macro')
    f1_weighted = f1_score(y_true=labels, y_pred=binary_preds, average='weighted')
    roc_auc = roc_auc_score(y_true=labels, y_score=sigmoid_preds, average='weighted')
    accuracy = accuracy_score(y_true=labels, y_pred=binary_preds)

    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'roc_auc': roc_auc,
        'accuracy': accuracy
    }

# --- Main Execution ---
def main():
    config = load_config()
    set_seed(config['seed'])

    # Load data
    df = pd.read_csv(config['preprocessed_data'])
    LABELS = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']
    X = df['full_text_cleaned'].astype(str)
    y = df[LABELS].values

    # Split data into train and test sets
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config['seed'])
    train_index, test_index = next(msss.split(X, y))
    
    # Further split training data into train and validation
    train_val_X, X_test = X.iloc[train_index], X.iloc[test_index]
    train_val_y, y_test = y[train_index], y[test_index]

    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=config['seed'])
    train_index_new, val_index = next(msss_val.split(train_val_X, train_val_y))

    X_train, X_val = train_val_X.iloc[train_index_new], train_val_X.iloc[val_index]
    y_train, y_val = train_val_y[train_index_new], train_val_y[val_index]

    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained(config['model_checkpoint'])
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=config['max_length'])
    val_encodings = tokenizer(list(X_val), truncation=True, padding=True, max_length=config['max_length'])
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=config['max_length'])

    # Create datasets
    train_dataset = MedicalDataset(train_encodings, y_train)
    val_dataset = MedicalDataset(val_encodings, y_val)
    test_dataset = MedicalDataset(test_encodings, y_test)

    # Model Initialization
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_checkpoint'], 
        num_labels=len(LABELS),
        problem_type="multi_label_classification"
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        report_to="none" # Disables wandb/tensorboard integration
    )

    # Trainer Initialization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    print("\nStarting BioBERT model fine-tuning...")
    trainer.train()

    # Evaluate on the test set
    print("\nEvaluating the fine-tuned model on the test set...")
    test_results = trainer.predict(test_dataset)
    
    print("\n--- Test Set Classification Report ---")
    y_true_test = test_dataset.labels
    test_preds = (1 / (1 + np.exp(-test_results.predictions)) > 0.5).astype(int)
    print(classification_report(y_true_test, test_preds, target_names=LABELS, zero_division=0))

    # Save the final metrics
    os.makedirs(config['reports_dir'], exist_ok=True)
    final_metrics = test_results.metrics
    with open(config['metrics_output_path'], 'w') as f:
        yaml.dump(final_metrics, f)
    print(f"Test metrics saved to {config['metrics_output_path']}")

if __name__ == "__main__":
    main()
