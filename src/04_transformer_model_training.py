"""
Version: v1.92
Author:
Date: March 2025

Purpose:
    - Train a transformer-based model (ClinicalBERT, Bio_ClinicalBERT, or similar) 
      for a classification task (e.g., disease progression prediction).
    - Showcase advanced features:
        * Mixed precision training
        * Gradient checkpointing
        * Gradient accumulation
        * Early stopping
        * Configurable hyperparameters (learning rate, epochs, warmup steps)
        * Advanced logging of training/validation metrics
    - Save best model checkpoints and final weights for deployment.

Input:
    - Tokenized/cleaned text data from a DuckDB or CSV
    - Label column for binary classification
    - Developer placeholders for file paths, hyperparameters, or API keys

Output:
    - Trained transformer model
    - Logs for training/validation performance
    - Optionally saved tokenizer/config for inference
"""

# ------------------------------------------------------------------------------
# External Dependencies (install if missing):
#   - python==3.12
#   - pandas==2.0.0
#   - numpy==1.24.0
#   - torch==2.0.0
#   - transformers==4.28.0 (or newer)
#   - scikit-learn==1.2.0
#   - duckdb==0.7.1 (optional, if loading from DuckDB)
#   - pyarrow==10.0.0
#   - matplotlib==3.5.0, seaborn==0.11.2 (optional, for advanced plotting)
#   - tqdm==4.64.0 (for progress bars)
# ------------------------------------------------------------------------------

import os
import sys
import logging
import duckdb
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    set_seed
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# <<< Default Placeholders, Add your own! >>>
DUCKDB_PATH = "<PATH_TO_INTEGRATED_DUCKDB>"         # e.g., "./mimic_integrated.duckdb"
VIEW_NAME = "target_patients_with_notes"             # e.g., table/view with text + label
TEXT_COLUMN = "cleaned_text"                         # text column name
LABEL_COLUMN = "progression_30d"                     # label column name
TRANSFORMER_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
OUTPUT_DIR = "<PATH_TO_MODEL_OUTPUT_DIR>"            # e.g., "./transformer_model"
MAX_LENGTH = 256                                     # max token length for truncation
BATCH_SIZE = 4                                       # effective batch per step
GRAD_ACCUM_STEPS = 4                                 # gradient accumulation -> total batch = BATCH_SIZE * GRAD_ACCUM_STEPS
LEARNING_RATE = 2e-5
EPOCHS = 3
WARMUP_RATIO = 0.1
SEED = 42
EARLY_STOPPING_PATIENCE = 2   # Stop if no improvement in F1 after X epochs
USE_GRADIENT_CHECKPOINTING = True
USE_MIXED_PRECISION = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------------------------
# GPU / Environment Setup
# ------------------------------------------------------------------------------
def setup_device():
    """
    Returns torch device (cuda if available) and logs GPU info if found.
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logging.info("CUDA-enabled GPU(s) detected: %d", device_count)
        for i in range(device_count):
            logging.info(" - GPU %d: %s", i, torch.cuda.get_device_name(i))
        return torch.device("cuda")
    else:
        logging.info("No GPU found. Using CPU.")
        return torch.device("cpu")

# ------------------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------------------
def load_data_from_duckdb(db_path, view=VIEW_NAME, text_col=TEXT_COLUMN, label_col=LABEL_COLUMN):
    """
    Connects to DuckDB, loads text+label from the specified view.
    Returns a pandas DataFrame.
    """
    if not os.path.exists(db_path):
        logging.error("DuckDB file not found at: %s", db_path)
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
# PyTorch Dataset
# ------------------------------------------------------------------------------
class TextClassificationDataset(Dataset):
    """
    Dataset that holds text/label pairs and uses a tokenizer to produce model inputs.
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
        label = float(self.labels[idx])
        # Hugging Face tokenization
        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Return a dict with the necessary inputs
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float)
        }

# ------------------------------------------------------------------------------
# Focal Loss (Optional) or Weighted BCE
# ------------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal loss for binary classification. 
    gamma > 1 penalizes well-classified examples, focusing on hard ones.
    alpha: weighting for positive class.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: shape [batch], targets: shape [batch]
        # Convert logits to probabilities using sigmoid
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * ((1 - pt) ** self.gamma)
        loss = focal_term * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ------------------------------------------------------------------------------
# Training + Evaluation
# ------------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, device, 
                    grad_accum_steps=GRAD_ACCUM_STEPS, 
                    use_mixed_precision=USE_MIXED_PRECISION, 
                    focal_loss=None, scaler=None):
    """
    Trains model for a single epoch, with optional gradient accumulation and mixed precision.
    If focal_loss is provided, that function overrides the default loss in the model.
    Returns the average loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    steps = 0

    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Mixed precision
        if use_mixed_precision and scaler is not None:
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None  # We'll handle the loss ourselves if focal
                )
                logits = outputs.logits.squeeze(-1)

                if focal_loss is not None:
                    loss = focal_loss(logits, labels)
                else:
                    # default BCE in huggingface is from `labels=...`
                    # We'll do it manually here:
                    loss_fn = nn.BCEWithLogitsLoss()
                    loss = loss_fn(logits, labels)
                
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            
            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        else:
            # No mixed precision
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None
            )
            logits = outputs.logits.squeeze(-1)

            if focal_loss is not None:
                loss = focal_loss(logits, labels)
            else:
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(logits, labels)

            loss = loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        total_loss += loss.item()
        steps += 1

    avg_loss = total_loss / steps if steps > 0 else 0.0
    return avg_loss

def evaluate(model, dataloader, device):
    """
    Evaluates model on a validation set. Returns tuple (accuracy, precision, recall, f1, auc).
    """
    model.eval()
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            preds_list.extend(probs.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    # Convert probabilities -> binary preds
    preds_binary = [1 if p >= 0.5 else 0 for p in preds_list]
    acc = accuracy_score(labels_list, preds_binary)
    prec = precision_score(labels_list, preds_binary, zero_division=0)
    rec = recall_score(labels_list, preds_binary, zero_division=0)
    f1 = f1_score(labels_list, preds_binary, zero_division=0)
    if len(np.unique(labels_list)) > 1:
        auc = roc_auc_score(labels_list, preds_list)
    else:
        auc = None

    return acc, prec, rec, f1, auc

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    """
    Main script:
        1. Set random seed for reproducibility
        2. Setup device
        3. Load data from DuckDB (or any other source)
        4. Split into train/val
        5. Create Dataset/Dataloader
        6. Setup model, possibly with gradient checkpointing
        7. Train with advanced features (mixed precision, focal loss, etc.)
        8. Evaluate each epoch, use early stopping
        9. Save best and final checkpoints
    """
    logging.info("=== Starting advanced transformer model training script ===")

    # 1. Fix random seeds
    set_seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 2. Setup device
    device = setup_device()

    # 3. Load data
    df = load_data_from_duckdb(DUCKDB_PATH, VIEW_NAME, TEXT_COLUMN, LABEL_COLUMN)
    if df.empty:
        logging.error("No data loaded. Exiting.")
        sys.exit(1)

    # 4. Train/Val split
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=SEED, stratify=df["label"])
    logging.info("Train size: %d, Val size: %d", len(train_df), len(val_df))

    # 5. Tokenizer & datasets
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
    train_dataset = TextClassificationDataset(train_df["text"].values, train_df["label"].values, tokenizer, MAX_LENGTH)
    val_dataset = TextClassificationDataset(val_df["text"].values, val_df["label"].values, tokenizer, MAX_LENGTH)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=BATCH_SIZE,
        drop_last=False
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=BATCH_SIZE,
        drop_last=False
    )

    # 6. Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_MODEL_NAME,
        num_labels=1  # Binary classification
    )
    # Gradient checkpointing to save memory
    if USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    model.to(device)

    # Optional focal loss (uncomment to use)
    # focal = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    focal = None

    # 7. Setup optimizer & scheduler
    total_steps = len(train_dataloader) * EPOCHS // GRAD_ACCUM_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Mixed precision scalar
    scaler = GradScaler() if USE_MIXED_PRECISION else None

    best_f1 = 0.0
    no_improve_count = 0
    best_model_path = None

    for epoch in range(EPOCHS):
        logging.info("Epoch %d / %d", epoch + 1, EPOCHS)
        # Train
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            use_mixed_precision=USE_MIXED_PRECISION,
            focal_loss=focal,
            scaler=scaler
        )
        logging.info("Train loss: %.4f", train_loss)

        # Evaluate
        acc, prec, rec, f1, auc = evaluate(model, val_dataloader, device)
        logging.info(
            "Validation -> Acc: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f, AUC: %s",
            acc, prec, rec, f1, f"{auc:.4f}" if auc else "N/A"
        )

        # Early stopping based on F1
        if f1 > best_f1:
            best_f1 = f1
            no_improve_count = 0

            # Save best model checkpoint
            best_model_path = os.path.join(OUTPUT_DIR, "best_model")
            os.makedirs(best_model_path, exist_ok=True)
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            logging.info("Saved new best model (F1=%.4f) to: %s", best_f1, best_model_path)

        else:
            no_improve_count += 1
            logging.info("No improvement in F1 for %d epoch(s).", no_improve_count)
            if no_improve_count >= EARLY_STOPPING_PATIENCE:
                logging.info("Early stopping triggered at epoch %d.", epoch + 1)
                break

    # Save final model
    final_model_path = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logging.info("Final model saved to: %s", final_model_path)

    logging.info("=== Transformer model training script finished successfully ===")


if __name__ == "__main__":
    main()