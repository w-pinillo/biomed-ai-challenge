import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def analyze_errors(y_true, y_pred, texts, labels, num_examples=3):
    """
    Performs a basic error analysis, printing per-label confusion matrices
    and example true positives, false positives, and false negatives.

    Args:
        y_true (np.array): True labels (one-hot encoded).
        y_pred (np.array): Predicted labels (binary, one-hot encoded).
        texts (list): List of original text corresponding to the samples.
        labels (list): List of label names.
        num_examples (int): Number of example texts to print for each category.
    """
    print("\n--- Starting Error Analysis ---")

    for i, label in enumerate(labels):
        print(f"\n--- Analysis for Label: {label} ---")
        
        true_label = y_true[:, i]
        pred_label = y_pred[:, i]

        # Confusion Matrix
        cm = confusion_matrix(true_label, pred_label)
        print("Confusion Matrix:")
        print(f"[[TN, FP]\n [FN, TP]]")
        print(cm)

        # Extract indices for TP, FP, FN
        tp_indices = np.where((true_label == 1) & (pred_label == 1))[0]
        fp_indices = np.where((true_label == 0) & (pred_label == 1))[0]
        fn_indices = np.where((true_label == 1) & (pred_label == 0))[0]

        # Print example cases
        print(f"\nExample True Positives ({len(tp_indices)} found):")
        for idx in tp_indices[:num_examples]:
            print(f"  - {texts[idx]}")
        if len(tp_indices) > num_examples:
            print(f"  ... ({len(tp_indices) - num_examples} more)")

        print(f"\nExample False Positives ({len(fp_indices)} found):")
        for idx in fp_indices[:num_examples]:
            print(f"  - {texts[idx]}")
        if len(fp_indices) > num_examples:
            print(f"  ... ({len(fp_indices) - num_examples} more)")

        print(f"\nExample False Negatives ({len(fn_indices)} found):")
        for idx in fn_indices[:num_examples]:
            print(f"  - {texts[idx]}")
        if len(fn_indices) > num_examples:
            print(f"  ... ({len(fn_indices) - num_examples} more)")

    print("\n--- Error Analysis Complete ---")
