"""
Version: v1.80
Author: 
Date: March 2025

Purpose:
    - Perform a final, comprehensive evaluation of both baseline and transformer models
    - Load the test set from a DuckDB or CSV
    - Apply baseline logistic regression model and transformer model to predict
    - Compare performance (accuracy, precision, recall, F1, ROC AUC)
    - Generate confusion matrices and optional plots (ROC, PR)
    - Provide consolidated results for analysis

Input:
    - Preprocessed test data from DuckDB or CSV
    - Baseline model (LogisticRegression) in joblib/pickle format
    - Transformer model checkpoint directory (Hugging Face format)
    - Developer placeholders for file paths, parameters, or API keys

Output:
    - Consolidated metrics for baseline and transformer
    - (Optional) confusion matrices, ROC, PR curves
    - Detailed logs of final performance
"""

# ------------------------------------------------------------------------------
# External Dependencies (Install if missing):
#   - python==3.12
#   - pandas==2.0.0
#   - numpy==1.24.0
#   - scikit-learn==1.2.0
#   - joblib==1.2.0
#   - duckdb==0.7.1 (optional, if using DuckDB to load data)
#   - transformers==4.28.0
#   - torch==2.0.0
#   - matplotlib==3.5.0, seaborn==0.11.2 (for plotting)
#   - tqdm==4.64.0
# ------------------------------------------------------------------------------

import os
import sys
import logging
import duckdb
import pandas as pd
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from torch.utils.data import Dataset, DataLoader, SequentialSampler

# <<< Default Placeholders, Add your own! >>>
DUCKDB_PATH = "<PATH_TO_INTEGRATED_DUCKDB>"              # e.g., "./mimic_integrated.duckdb"
VIEW_NAME = "target_patients_with_notes_test"             # e.g., test data with text + label
TEXT_COLUMN = "cleaned_text"                              # e.g., "cleaned_text"
LABEL_COLUMN = "progression_30d"                          # e.g., "progression_30d"
BASELINE_MODEL_PATH = "<PATH_TO_BASELINE_MODEL>"          # e.g., "./data_processed/models/baseline_logreg.pkl"
TFIDF_VECTORIZER_PATH = "<PATH_TO_TFIDF_VECTORIZER>"      # e.g., "./data_processed/features/tfidf_vectorizer.pkl"
TRANSFORMER_MODEL_DIR = "<PATH_TO_TRANSFORMER_BEST_MODEL>"# e.g., "./transformer_model/best_model"
OUTPUT_PLOTS_DIR = "<PATH_TO_PLOTS_OUTPUT>"               # e.g., "./results_plots"
MAX_LENGTH = 256
BATCH_SIZE = 8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------------------
def load_test_data_from_duckdb(db_path, view=VIEW_NAME, text_col=TEXT_COLUMN, label_col=LABEL_COLUMN):
    """
    Loads test data from DuckDB. Returns a DataFrame with columns: 'text' and 'label'.
    """
    if not os.path.exists(db_path):
        logging.error("DuckDB database not found at %s", db_path)
        return pd.DataFrame()

    con = duckdb.connect(db_path, read_only=True)
    query = f"""
        SELECT {text_col} as text, {label_col} as label
        FROM {view}
        WHERE {text_col} IS NOT NULL
          AND {label_col} IS NOT NULL
    """
    df = con.execute(query).df()
    con.close()
    logging.info("Loaded %d rows from DuckDB view '%s'.", len(df), view)
    return df

# ------------------------------------------------------------------------------
# Baseline Model Evaluation
# ------------------------------------------------------------------------------
def evaluate_baseline_model(df, text_col, label_col,
                            model_path=BASELINE_MODEL_PATH,
                            vectorizer_path=TFIDF_VECTORIZER_PATH):
    """
    Loads baseline logistic regression model and TF-IDF vectorizer from disk.
    Transforms text, gets predictions, computes metrics.
    Returns (metrics_dict, predictions, probabilities).
    """
    if not os.path.exists(model_path):
        logging.error("Baseline model file not found: %s", model_path)
        return {}, [], []

    if not os.path.exists(vectorizer_path):
        logging.error("TF-IDF vectorizer file not found: %s", vectorizer_path)
        return {}, [], []

    # Load model + vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Transform text
    X_text = df[text_col].fillna("").astype(str).tolist()
    X_tfidf = vectorizer.transform(X_text)

    # Predict
    y_true = df[label_col].values
    y_pred = model.predict(X_tfidf)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_tfidf)[:, 1]
    else:
        # fallback if no predict_proba
        y_proba = np.array(y_pred, dtype=float)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_proba, name="Baseline Model")
    return metrics, y_pred, y_proba

# ------------------------------------------------------------------------------
# Transformer Model Evaluation
# ------------------------------------------------------------------------------
class TransformerEvalDataset(Dataset):
    """
    Simple dataset for text classification with a Hugging Face tokenizer.
    """
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": label
        }

def evaluate_transformer_model(df, text_col, label_col,
                               model_dir=TRANSFORMER_MODEL_DIR,
                               batch_size=BATCH_SIZE):
    """
    Loads a transformer model from 'model_dir', evaluates on the provided DataFrame.
    Returns (metrics_dict, predictions, probabilities).
    """
    if not os.path.exists(model_dir):
        logging.error("Transformer model directory not found: %s", model_dir)
        return {}, [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Create dataset + dataloader
    texts = df[text_col].fillna("").astype(str).tolist()
    labels = df[label_col].astype(float).values
    dataset = TransformerEvalDataset(texts, labels, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        drop_last=False
    )

    preds = []
    probs = []
    y_true = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            prob = torch.sigmoid(logits)
            preds_batch = (prob >= 0.5).long().cpu().numpy()
            preds.extend(preds_batch.tolist())
            probs.extend(prob.cpu().numpy().tolist())
            y_true.extend(labels.tolist())

    # Compute metrics
    metrics = compute_metrics(np.array(y_true), np.array(preds), np.array(probs), name="Transformer Model")
    return metrics, preds, probs

# ------------------------------------------------------------------------------
# Metrics and Plots
# ------------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba, name="Model"):
    """
    Computes standard metrics: accuracy, precision, recall, f1, auc.
    Returns dict of results.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from numpy import unique as np_unique

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    if len(np_unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_proba)
    else:
        auc = None

    logging.info("%s -> ACC: %.4f, PREC: %.4f, REC: %.4f, F1: %.4f, AUC: %s",
                 name, acc, prec, rec, f1, f"{auc:.4f}" if auc else "N/A")

    return {
        "model_name": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc
    }

def plot_confusion_matrix(y_true, y_pred, title, output_dir=OUTPUT_PLOTS_DIR):
    """
    Saves a confusion matrix plot for y_true vs. y_pred.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{title.replace(' ','_').lower()}_cm.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved confusion matrix plot: %s", out_path)

def plot_roc_curve(y_true, y_proba, title, output_dir=OUTPUT_PLOTS_DIR):
    """
    Plots and saves ROC curve. If only one class, skip.
    """
    if len(np.unique(y_true)) < 2:
        logging.warning("Cannot plot ROC: only one class in y_true.")
        return
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{title.replace(' ','_').lower()}_roc.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved ROC curve: %s", out_path)

def plot_precision_recall(y_true, y_proba, title, output_dir=OUTPUT_PLOTS_DIR):
    """
    Plots and saves precision-recall curve. If only one class, skip.
    """
    if len(np.unique(y_true)) < 2:
        logging.warning("Cannot plot PR: only one class in y_true.")
        return
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="best")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{title.replace(' ','_').lower()}_pr.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved Precision-Recall curve: %s", out_path)

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    """
    Main function:
        1. Load final test set from DuckDB
        2. Evaluate baseline logistic regression + TF-IDF
        3. Evaluate transformer model
        4. Compare and produce final logs + plots
    """
    logging.info("=== Starting final model evaluation script ===")

    # 1. Load test data
    df_test = load_test_data_from_duckdb(DUCKDB_PATH, VIEW_NAME, TEXT_COLUMN, LABEL_COLUMN)
    if df_test.empty:
        logging.error("No test data loaded. Exiting.")
        sys.exit(1)

    # 2. Evaluate Baseline
    baseline_metrics, baseline_preds, baseline_proba = evaluate_baseline_model(
        df_test,
        text_col="text",
        label_col="label",
        model_path=BASELINE_MODEL_PATH,
        vectorizer_path=TFIDF_VECTORIZER_PATH
    )

    # 3. Evaluate Transformer
    transformer_metrics, transformer_preds, transformer_proba = evaluate_transformer_model(
        df_test,
        text_col="text",
        label_col="label",
        model_dir=TRANSFORMER_MODEL_DIR,
        batch_size=BATCH_SIZE
    )

    # 4. Compare & produce final logs
    # Display classification reports for both if needed
    y_true = df_test["label"].values

    # Baseline plots
    if baseline_metrics:
        plot_confusion_matrix(y_true, baseline_preds, "Baseline Confusion Matrix")
        plot_roc_curve(y_true, baseline_proba, "Baseline ROC Curve")
        plot_precision_recall(y_true, baseline_proba, "Baseline PR Curve")

    # Transformer plots
    if transformer_metrics:
        plot_confusion_matrix(y_true, transformer_preds, "Transformer Confusion Matrix")
        plot_roc_curve(y_true, transformer_proba, "Transformer ROC Curve")
        plot_precision_recall(y_true, transformer_proba, "Transformer PR Curve")

    # Print final summary
    logging.info("=== Final Results ===")
    if baseline_metrics:
        logging.info("Baseline: %s", baseline_metrics)
    if transformer_metrics:
        logging.info("Transformer: %s", transformer_metrics)

    logging.info("=== Final model evaluation completed successfully ===")

if __name__ == "__main__":
    main()