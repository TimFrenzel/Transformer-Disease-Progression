"""
Version: v1.10
Author: 
Date: March 2025

Purpose:
    - Generate feature sets for downstream modeling, focusing on:
        1. TF-IDF vectorization of clinical text
        2. Optional basic embedding generation (if needed, see placeholders)
        3. Structured feature extraction (demographic, numeric)
    - Implement a baseline Logistic Regression model:
        1. Train/test split or cross-validation
        2. Hyperparameter tuning (grid search)
        3. Performance metrics (accuracy, precision, recall, F1, ROC, PR)
        4. Feature importance analysis
    - Output model artifacts and feature files for further evaluation

Input:
    - Preprocessed/cleaned text data (e.g., from DuckDB or CSV)
    - Optional structured data (demographics, numeric fields)
    - Developer placeholders for file paths, parameters, or API keys

Output:
    - TF-IDF vectorizer and sparse matrices saved (if applicable)
    - Trained baseline logistic regression model in a serialized format
    - Performance metrics and feature importance plots/CSVs
"""

# ------------------------------------------------------------------------------
# External Dependencies (Install if missing):
#   - python==3.12
#   - pandas==2.0.0 (or newer)
#   - numpy==1.24.0 (or newer)
#   - scikit-learn==1.2.0 (or newer)
#   - duckdb==0.7.1 (or newer)  # if loading from a DuckDB
#   - pyarrow==10.0.0 (or newer)
#   - matplotlib==3.5.0 (or newer)
#   - seaborn==0.11.2 (or newer)
#   - joblib==1.2.0 (for saving/loading model)
# ------------------------------------------------------------------------------

import os
import logging
import pandas as pd
import numpy as np
import duckdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedKFold)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, precision_recall_curve)
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# <<< Default Placeholders, Add your own!
DUCKDB_PATH = "<PATH_TO_INTEGRATED_DUCKDB>"  # e.g., "./mimic_integrated.duckdb"
FEATURES_OUTPUT_DIR = "<PATH_TO_FEATURES_DIR>"  # e.g., "./data_processed/features"
MODEL_OUTPUT_DIR = "<PATH_TO_MODELS_DIR>"       # e.g., "./data_processed/models"
PLOTS_OUTPUT_DIR = "<PATH_TO_PLOTS_DIR>"        # e.g., "./data_processed/plots"
SELECTED_VIEW = "target_patients_with_notes"    # view name in DuckDB for text + label
TEXT_COLUMN = "cleaned_text"                    # column in the DuckDB view for text
LABEL_COLUMN = "progression_30d"                # column for binary label, adjust as needed
# E.g., "progression_30d" means sepsis/pneumonia progression within 30 days

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
def load_data_from_duckdb(db_path, view_name=SELECTED_VIEW, text_col=TEXT_COLUMN, label_col=LABEL_COLUMN):
    """
    Connects to DuckDB, reads data from the specified view, and returns a DataFrame with text and label.
    Adjust as needed for additional structured features.
    """
    if not os.path.exists(db_path):
        logging.error("DuckDB file not found at: %s", db_path)
        return pd.DataFrame()

    logging.info("Loading data from DuckDB view '%s' in %s", view_name, db_path)
    con = duckdb.connect(db_path, read_only=True)
    # Example minimal columns:
    query = f"""
        SELECT {text_col}, {label_col}
        FROM {view_name}
        WHERE {text_col} IS NOT NULL
              AND {label_col} IS NOT NULL
    """
    df = con.execute(query).df()
    con.close()
    logging.info("Loaded %d rows from DuckDB.", len(df))
    return df

def split_data(df, text_col=TEXT_COLUMN, label_col=LABEL_COLUMN, test_size=0.2, random_state=42):
    """
    Splits the DataFrame into training and test sets.
    Returns (X_train, X_test, y_train, y_test).
    """
    X = df[text_col].values
    y = df[label_col].values
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plots and saves a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    out_path = os.path.join(PLOTS_OUTPUT_DIR, f"{title.replace(' ', '_').lower()}.png")
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved %s plot: %s", title, out_path)

def plot_roc_curve_proba(y_true, y_proba, title="ROC Curve"):
    """
    Plots ROC curve given true labels and predicted probabilities for the positive class.
    """
    if len(np.unique(y_true)) < 2:
        logging.warning("Cannot plot ROC curve; only one class present.")
        return
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    out_path = os.path.join(PLOTS_OUTPUT_DIR, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved %s plot: %s", title, out_path)

def plot_precision_recall_curve(y_true, y_proba, title="Precision-Recall Curve"):
    """
    Plots precision-recall curve given true labels and predicted probabilities.
    """
    if len(np.unique(y_true)) < 2:
        logging.warning("Cannot plot PR curve; only one class present.")
        return
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score_safe(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="best")
    out_path = os.path.join(PLOTS_OUTPUT_DIR, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved %s plot: %s", title, out_path)

def average_precision_score_safe(y_true, y_proba):
    """
    Returns average precision score if feasible, otherwise returns 0.0.
    """
    from sklearn.metrics import average_precision_score
    if len(np.unique(y_true)) < 2:
        return 0.0
    return average_precision_score(y_true, y_proba)

# ------------------------------------------------------------------------------
# Feature Engineering: TF-IDF
# ------------------------------------------------------------------------------
def build_tfidf_features(X_train, X_test, max_features=20000, ngram_range=(1,2), min_df=2, max_df=0.9):
    """
    Fits a TF-IDF vectorizer on X_train and transforms both train/test sets.
    Saves vectorizer if needed. Returns (X_train_tfidf, X_test_tfidf, vectorizer).
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words='english'
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer

def save_tfidf_vectorizer(vectorizer, output_dir=FEATURES_OUTPUT_DIR, name="tfidf_vectorizer.pkl"):
    """
    Saves the fitted TF-IDF vectorizer to disk using joblib.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, name)
    joblib.dump(vectorizer, out_path)
    logging.info("Saved TF-IDF vectorizer to: %s", out_path)

# ------------------------------------------------------------------------------
# Baseline Model: Logistic Regression
# ------------------------------------------------------------------------------
def train_baseline_model(X_train, y_train, param_grid=None, cv_folds=5, scoring='f1'):
    """
    Trains a logistic regression model with optional grid search for hyperparameters.
    Returns the best estimator (model) and the grid search object.
    """
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['liblinear']
        }
    logging.info("Starting logistic regression training with grid search.")
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=skf, scoring=scoring, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    logging.info("Best params: %s, best score=%.4f", grid_search.best_params_, grid_search.best_score_)
    return best_model, grid_search

def save_model(model, model_dir=MODEL_OUTPUT_DIR, name="baseline_logreg.pkl"):
    """
    Saves the trained model to disk.
    """
    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, name)
    joblib.dump(model, out_path)
    logging.info("Saved logistic regression model to: %s", out_path)

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def main():
    """
    Main function for feature engineering and baseline model training:
     1. Load data from DuckDB
     2. Split into train/test
     3. Build TF-IDF features
     4. Train logistic regression baseline
     5. Evaluate on test set
     6. Save artifacts (vectorizer, model, plots)
    """
    logging.info("=== Starting feature engineering & baseline model script ===")

    # 1. Load data
    df = load_data_from_duckdb(DUCKDB_PATH, SELECTED_VIEW, TEXT_COLUMN, LABEL_COLUMN)
    if df.empty:
        logging.error("No data loaded. Exiting.")
        return
    logging.info("Data shape: %s", df.shape)

    # 2. Split
    X_train, X_test, y_train, y_test = split_data(df)
    logging.info("Train shape: %s, Test shape: %s", X_train.shape[0], X_test.shape[0])

    # 3. Build TF-IDF features
    X_train_tfidf, X_test_tfidf, tfidf_vec = build_tfidf_features(X_train, X_test)
    save_tfidf_vectorizer(tfidf_vec)

    logging.info("TF-IDF Train shape: %s, TF-IDF Test shape: %s",
                 X_train_tfidf.shape, X_test_tfidf.shape)

    # 4. Train baseline logistic regression
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'solver': ['liblinear', 'saga']
    }
    best_model, gs_obj = train_baseline_model(X_train_tfidf, y_train, param_grid=param_grid)

    # 5. Evaluate on test set
    y_pred = best_model.predict(X_test_tfidf)
    y_proba = best_model.predict_proba(X_test_tfidf)[:, 1] if hasattr(best_model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    if y_proba is not None and len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = None

    logging.info("Test Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f, AUC=%s",
                 acc, prec, rec, f1, f"{auc:.4f}" if auc else "N/A")

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, "Baseline_Confusion_Matrix")

    # ROC curve
    if y_proba is not None:
        plot_roc_curve_proba(y_test, y_proba, "Baseline_ROC_Curve")
        plot_precision_recall_curve(y_test, y_proba, "Baseline_Precision_Recall_Curve")

    # Classification report
    report = classification_report(y_test, y_pred, zero_division=0)
    logging.info("Classification Report:\n%s", report)

    # 6. Save model
    save_model(best_model, MODEL_OUTPUT_DIR, "baseline_logreg.pkl")

    logging.info("=== Feature engineering & baseline model script completed successfully ===")

if __name__ == "__main__":
    main()
