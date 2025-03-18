"""
Version: v1.64
Date: March 2025

Purpose:
    - Configure and verify the Python environment for data science tasks
    - Perform initial data exploration on MIMIC-IV note files (discharge, discharge_detail, radiology, radiology_detail) 
      and FHIR NDJSON resources (MimicCondition, MimicConditionED, MimicEncounter, MimicEncounterED, MimicPatient)
    - Convert a large CSV (e.g., discharge.csv.gz) to multiple Parquet files using chunk-based processing
    - Demonstrate basic SQL-based EDA using DuckDB (schema extraction, row counts, sample queries)
    - Produce a brief timeline visualization for selected patients
    - Produce relevant, well-designed EDA plots for note files and FHIR resources

Input:
    - Large MIMIC-IV CSV/Parquet files (discharge, discharge_detail, radiology, radiology_detail)
    - FHIR NDJSON files (MimicCondition, MimicEncounter, MimicPatient, etc.)

Output:
    - Environment checks (Python version, GPU availability)
    - CSV to Parquet chunked outputs
    - Basic EDA results (schemas, row counts, distribution plots)
    - Timeline visualization for selected patients
    - Saved PNG images for EDA plots
"""

# ------------------------------------------------------------------------------
# External Dependencies (Install if missing):
#   - python==3.12
#   - pandas==2.0.0 (or newer)
#   - duckdb==0.7.1 (or newer)
#   - pyarrow==10.0.0 (or newer)
#   - torch==2.0.0 (or newer)          # If using PyTorch GPU checks
#   - matplotlib==3.5.0 (or newer)
#   - seaborn==0.11.2 (or newer)
#   - orjson / json / gzip (for NDJSON, optional)
# ------------------------------------------------------------------------------

import sys
import os
import logging
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# If NDJSON reading is required, these might be necessary:
import gzip
import json
# from orjson import loads  # Alternative if you prefer orjson

# <<< Default Placeholders, Add your own!
DATA_DIR = "<PATH_TO_MIMIC_IV_DATA>"  # e.g., "/data/mimic4_db/note"
FHIR_DIR = "<PATH_TO_MIMIC_FHIR_DATA>"  # e.g., "/data/mimic4_db/FHIR"
PARQUET_DIR = "<OUTPUT_PARQUET_DIR>"    # e.g., "/data/mimic4_db/PARQUET"
CHUNK_SIZE = 100_000                    # Number of rows per CSV chunk
PLOTS_DIR = "./plots"                   # Where to save generated EDA plots
DUCKDB_MEMORY_LIMIT = "4GB"             # Memory limit for DuckDB if needed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def check_python_version(required_major=3, required_minor=8):
    """
    Checks if the current Python version meets a minimum requirement.
    Raises SystemExit if below the requirement.
    """
    version_info = sys.version_info
    if (version_info.major < required_major) or \
       (version_info.major == required_major and version_info.minor < required_minor):
        logging.error(
            "Python %d.%d+ is required. Found Python %d.%d",
            required_major, required_minor, version_info.major, version_info.minor
        )
        sys.exit(1)
    else:
        logging.info("Python version is sufficient: %d.%d.%d",
                     version_info.major, version_info.minor, version_info.micro)

def check_gpu_availability():
    """
    Checks if a CUDA-enabled GPU is available via PyTorch.
    Logs GPU details if available; warns if PyTorch is missing.
    """
    try:
        import torch
    except ImportError:
        logging.warning("PyTorch is not installed. Skipping GPU check.")
        return

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            logging.info("GPU %d: %s", i, device_name)
        logging.info("CUDA-enabled GPU(s) detected.")
    else:
        logging.info("No CUDA-enabled GPU found. Running on CPU.")

def convert_csv_to_parquet_chunks(csv_path, output_dir, chunk_size=CHUNK_SIZE):
    """
    Converts a large CSV file to multiple Parquet files in 'output_dir' 
    by reading chunks of 'chunk_size'. Logs progress for each chunk.
    """
    if not os.path.exists(csv_path):
        logging.error("CSV file not found: %s", csv_path)
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chunk_index = 0
    try:
        for chunk_df in pd.read_csv(csv_path, chunksize=chunk_size, compression='infer'):
            out_parquet = os.path.join(output_dir, f"discharge_chunk_{chunk_index}.parquet")
            chunk_df.to_parquet(out_parquet, index=False)
            logging.info("Wrote chunk %d to %s, shape=%s", chunk_index, out_parquet, chunk_df.shape)
            chunk_index += 1
    except Exception as e:
        logging.error("Failed to convert CSV to Parquet in chunks: %s", e)

    logging.info("Total chunks created: %d", chunk_index)

def setup_duckdb_and_extract_schema(csv_path):
    """
    Sets up an in-memory DuckDB connection for schema extraction only.
    Prints row count and schema for the provided CSV file.
    """
    if not os.path.exists(csv_path):
        logging.warning("File not found: %s. Skipping schema extraction.", csv_path)
        return

    con = duckdb.connect(database=":memory:")
    # Optional memory limit
    con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}'")

    try:
        query = f"""
            DESCRIBE (SELECT * FROM read_csv_auto('{csv_path}') LIMIT 0)
        """
        schema_df = con.execute(query).df()
        logging.info("Schema for %s:\n%s", os.path.basename(csv_path), schema_df)

        count_query = f"SELECT COUNT(*) FROM read_csv_auto('{csv_path}')"
        row_count = con.execute(count_query).fetchone()[0]
        logging.info("Row count for %s: %d", os.path.basename(csv_path), row_count)
    except Exception as e:
        logging.error("DuckDB schema extraction failed for %s: %s", csv_path, e)

    con.close()

def load_fhir_sample(fhir_path, limit=1000):
    """
    Loads a subset (up to 'limit') from a NDJSON.gz FHIR file.
    Returns the data as a list of JSON records or an empty list if file not found.
    """
    if not os.path.exists(fhir_path):
        logging.warning("FHIR file not found: %s", fhir_path)
        return []

    records = []
    try:
        with gzip.open(fhir_path, 'rt', encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                record = json.loads(line)
                records.append(record)
    except Exception as e:
        logging.error("Failed to load FHIR NDJSON from %s: %s", fhir_path, e)
        return []

    logging.info("Loaded %d records from %s", len(records), os.path.basename(fhir_path))
    return records

def plot_timeline_for_selected_patients(df, patient_id_col="subject_id", time_col="charttime"):
    """
    Produces a brief timeline visualization for selected patients from a DataFrame 
    that includes columns for patient_id and time. 
    This function expects datetime-like data in 'time_col'.
    """
    if df.empty or patient_id_col not in df.columns or time_col not in df.columns:
        logging.warning("Timeline data not suitable for plotting.")
        return

    # Convert time_col to datetime if not already
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[patient_id_col, time_col])

    # We only pick a handful of patients for demonstration
    unique_patients = df[patient_id_col].unique()
    sample_patients = unique_patients[:5]  # pick up to 5 patients

    # Create a plot for each patient showing events over time
    plt.figure(figsize=(10, 6))
    for idx, patient_id in enumerate(sample_patients):
        patient_df = df[df[patient_id_col] == patient_id].copy()
        patient_df.sort_values(time_col, inplace=True)
        # Example timeline as x=charttime, y=some index
        plt.plot(patient_df[time_col], [idx]*len(patient_df), marker='o', label=f"Patient {patient_id}")

    plt.legend()
    plt.title("Timeline of Selected Patients")
    plt.xlabel("Time")
    plt.ylabel("Patient Index")
    out_png = os.path.join(PLOTS_DIR, "timeline_selected_patients.png")
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    plt.savefig(out_png)
    plt.close()
    logging.info("Saved timeline plot for selected patients: %s", out_png)

# --- EDA Plotting Functions ---
def plot_missing_values(df, file_label):
    """
    Plots missing value counts for each column in df as a horizontal bar chart.
    """
    missing_counts = df.isna().sum().sort_values(ascending=False)
    if missing_counts.sum() == 0:
        logging.info("No missing values in %s data. Skipping plot.", file_label)
        return

    plt.figure(figsize=(8, 6))
    missing_counts.plot(kind='barh', color='skyblue')
    plt.xlabel("Missing Count")
    plt.title(f"Missing Values in {file_label}")
    out_path = os.path.join(PLOTS_DIR, f"{file_label}_missing_values.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logging.info("Saved missing values plot: %s", out_path)

def plot_text_length_distribution(df, text_col="text", file_label="discharge"):
    """
    Creates a histogram of text lengths for a given text column in df.
    """
    if text_col not in df.columns:
        logging.warning("No '%s' column found in %s data. Skipping text length plot.", text_col, file_label)
        return

    df["text_length"] = df[text_col].apply(lambda x: len(x) if isinstance(x, str) else 0)
    plt.figure(figsize=(8, 6))
    sns.histplot(df["text_length"], bins=30, color='teal', edgecolor='black')
    plt.title(f"Text Length Distribution - {file_label}")
    plt.xlabel("Text Length (chars)")
    plt.ylabel("Frequency")
    out_path = os.path.join(PLOTS_DIR, f"{file_label}_text_length_hist.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logging.info("Saved text length distribution plot: %s", out_path)

def plot_fhir_resource_type(records, file_label="MimicCondition"):
    """
    Given a list of FHIR records (dicts), plot the distribution of resourceType fields.
    """
    if not records:
        logging.warning("No records for %s, skipping resourceType plot.", file_label)
        return

    resource_types = []
    for rec in records:
        rt = rec.get("resourceType", "Unknown")
        resource_types.append(rt)

    if not resource_types:
        logging.warning("No resourceType found in %s records, skipping plot.", file_label)
        return

    sr = pd.Series(resource_types).value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sr.index, y=sr.values, palette="viridis")
    plt.title(f"ResourceType Distribution - {file_label}")
    plt.xlabel("ResourceType")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    out_path = os.path.join(PLOTS_DIR, f"{file_label}_resourceType_distribution.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logging.info("Saved FHIR resourceType distribution plot: %s", out_path)

# --- Main Routine ---
def main():
    """
    Main execution function:
    - Verifies Python environment & GPU
    - Converts large CSV to Parquet in chunks
    - Uses DuckDB to do a quick schema extraction & row count
    - Performs minimal EDA on note files (discharge, etc.) and FHIR data
    - Produces 3 relevant EDA plots + timeline plot demonstration
    """
    logging.info("=== Starting environment & data exploration script ===")

    # 1. Environment Checks
    check_python_version(required_major=3, required_minor=8)
    check_gpu_availability()

    # 2. CSV to Parquet Conversion (discharge.csv.gz as example)
    csv_file = os.path.join(DATA_DIR, "discharge.csv.gz")  # <<< Adjust as needed
    convert_csv_to_parquet_chunks(csv_file, PARQUET_DIR, chunk_size=CHUNK_SIZE)

    # 3. Setup DuckDB & Extract Schema (just for discharge as example)
    setup_duckdb_and_extract_schema(csv_file)

    # 4. Minimal EDA on MIMIC-IV note files
    #    We'll read small samples from each file to demonstrate
    note_files = {
        "discharge": os.path.join(DATA_DIR, "discharge.csv.gz"),
        "discharge_detail": os.path.join(DATA_DIR, "discharge_detail.csv.gz"),
        "radiology": os.path.join(DATA_DIR, "radiology.csv.gz"),
        "radiology_detail": os.path.join(DATA_DIR, "radiology_detail.csv.gz")
    }

    for label, path in note_files.items():
        if not os.path.exists(path):
            logging.warning("%s file not found, skipping EDA for that file.", label)
            continue
        # Load a small sample
        try:
            sample_df = pd.read_csv(path, nrows=2000, compression='infer')
            logging.info("Loaded sample from %s with shape %s", label, sample_df.shape)

            # Quick missing-value plot
            plot_missing_values(sample_df, file_label=label)

            # Plot text length distribution if "text" column found
            if "text" in sample_df.columns:
                plot_text_length_distribution(sample_df, text_col="text", file_label=label)

            # Optional timeline plot if columns exist
            if "subject_id" in sample_df.columns and "charttime" in sample_df.columns:
                plot_timeline_for_selected_patients(sample_df, "subject_id", "charttime")

        except Exception as e:
            logging.error("Failed to process %s data for EDA: %s", label, e)

    # 5. Minimal EDA on FHIR NDJSON resources 
    #    We'll load a small sample from each and plot resourceType distribution
    fhir_files = {
        "MimicCondition": os.path.join(FHIR_DIR, "MimicCondition.ndjson.gz"),
        "MimicConditionED": os.path.join(FHIR_DIR, "MimicConditionED.ndjson.gz"),
        "MimicEncounter": os.path.join(FHIR_DIR, "MimicEncounter.ndjson.gz"),
        "MimicEncounterED": os.path.join(FHIR_DIR, "MimicEncounterED.ndjson.gz"),
        "MimicPatient": os.path.join(FHIR_DIR, "MimicPatient.ndjson.gz")
    }

    for label, path in fhir_files.items():
        recs = load_fhir_sample(path, limit=1000)
        plot_fhir_resource_type(recs, file_label=label)

    logging.info("=== Data exploration script finished successfully ===")

if __name__ == "__main__":
    main()