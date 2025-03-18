"""
Version: v1.4
Author: 
Date: March 2025

Purpose:
    - Integrate MIMIC-IV note data with MIMIC-IV on FHIR data into a DuckDB database
    - Focus on sepsis and pneumonia diagnoses (plus relevant patients)
    - Preprocess clinical text: remove boilerplate, de-identify, fix OCR errors, tokenize
    - Create disease progression labels, if desired
    - Provide a unified data source (DuckDB) for downstream modeling

Input:
    - MIMIC-IV note files (e.g., discharge.csv.gz, radiology.csv.gz) 
    - MIMIC-IV on FHIR NDJSON files (MimicPatient, MimicEncounter, MimicCondition)
    - Developer placeholders for file paths and parameters

Output:
    - A DuckDB database with integrated tables (patients, encounters, conditions, discharge_notes)
    - Cleaned, preprocessed text stored in the database or external files
    - Optional progression labels or relevant metadata for sepsis/pneumonia patients
"""

# ------------------------------------------------------------------------------
# External Dependencies (Install if missing):
#   - python==3.12
#   - pandas==2.0.0 (or newer)
#   - duckdb==0.7.1 (or newer)
#   - pyarrow==10.0.0 (or newer)
#   - numpy==1.24.0 (or newer)
#   - nltk==3.8 (for stopwords, if used)
#   - re / regex libraries (stdlib)
#   - logging (stdlib)
# ------------------------------------------------------------------------------

import os
import re
import sys
import logging
import gzip
import json
import gc
import duckdb
import pandas as pd
import numpy as np

# If you need NLTK stopwords:
# import nltk
# from nltk.corpus import stopwords

# <<< Default Placeholders, Add your own!
FHIR_DIR = "<PATH_TO_FHIR_DATA>"        # e.g., "/data/mimic4_db/FHIR"
NOTES_DIR = "<PATH_TO_NOTE_DATA>"       # e.g., "/data/mimic4_db/note"
DB_OUTPUT_PATH = "<PATH_TO_INTEGRATED_DUCKDB>"  # e.g., "./mimic_integrated.duckdb"
TEMP_DIR = "<PATH_TO_TEMP_DIR>"         # e.g., "./temp_data"
MEMORY_LIMIT = "8GB"
THREADS = 4

# If you plan to store preprocessed text externally:
PREPROCESSED_DIR = "<PATH_TO_PREPROCESSED_TEXT>"  # e.g., "./data_processed/preprocessed_notes"

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------------------------
# Helper: Initialize DuckDB
# ------------------------------------------------------------------------------
def initialize_duckdb(db_path, memory_limit=MEMORY_LIMIT, threads=THREADS, temp_dir=TEMP_DIR):
    """
    Initializes a DuckDB connection with the specified memory limit and temp directory.
    Returns the connection object.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA threads={threads}")
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
        con.execute(f"PRAGMA temp_directory='{temp_dir}'")
    logging.info("DuckDB initialized at: %s", db_path)
    return con

# ------------------------------------------------------------------------------
# Data Integration: FHIR Data
# ------------------------------------------------------------------------------
def process_patient_data(con, patient_file):
    """
    Reads MIMIC patient data from a NDJSON.gz file, extracts relevant fields,
    and inserts into a 'patients' table in DuckDB.
    """
    logging.info("Processing patient data from %s", os.path.basename(patient_file))
    con.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            fhir_id VARCHAR,
            subject_id VARCHAR,
            gender VARCHAR,
            birth_date DATE
        )
    """)
    existing_count = con.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    if existing_count > 0:
        logging.info("patients table already has %d records, skipping reload.", existing_count)
        return

    batch_size = 10000
    records_buffer = []
    records_processed = 0

    if not os.path.exists(patient_file):
        logging.warning("Patient file not found: %s. Skipping.", patient_file)
        return

    with gzip.open(patient_file, 'rt', encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fhir_id = rec.get("id")
            gender = rec.get("gender", "")
            birth_date = rec.get("birthDate", None)
            subject_id = None
            for ident in rec.get("identifier", []):
                if ident.get("system") == "http://mimic.mit.edu/fhir/mimic/identifier/patient":
                    subject_id = ident.get("value")
                    break

            if fhir_id and subject_id:
                records_buffer.append({
                    "fhir_id": fhir_id,
                    "subject_id": subject_id,
                    "gender": gender,
                    "birth_date": birth_date
                })
            records_processed += 1
            if len(records_buffer) >= batch_size:
                df_batch = pd.DataFrame(records_buffer)
                con.execute("INSERT INTO patients SELECT * FROM df_batch")
                records_buffer = []
                gc.collect()
        # final flush
        if records_buffer:
            df_batch = pd.DataFrame(records_buffer)
            con.execute("INSERT INTO patients SELECT * FROM df_batch")

    final_count = con.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    logging.info("Loaded %d patient records into DuckDB (unique inserted: %d).",
                 records_processed, final_count)

def process_encounter_data(con, encounter_file):
    """
    Reads MIMIC encounter data from NDJSON.gz, loads into 'encounters' DuckDB table.
    """
    logging.info("Processing encounter data from %s", os.path.basename(encounter_file))
    con.execute("""
        CREATE TABLE IF NOT EXISTS encounters (
            encounter_fhir_id VARCHAR,
            hadm_id VARCHAR,
            patient_fhir_id VARCHAR,
            start_time TIMESTAMP,
            end_time TIMESTAMP
        )
    """)
    existing_count = con.execute("SELECT COUNT(*) FROM encounters").fetchone()[0]
    if existing_count > 0:
        logging.info("encounters table already has %d records, skipping reload.", existing_count)
        return

    batch_size = 10000
    records_buffer = []
    records_processed = 0

    if not os.path.exists(encounter_file):
        logging.warning("Encounter file not found: %s. Skipping.", encounter_file)
        return

    with gzip.open(encounter_file, 'rt', encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fhir_id = rec.get("id")
            subj_ref = rec.get("subject", {}).get("reference", None)
            if subj_ref and subj_ref.startswith("Patient/"):
                patient_fhir_id = subj_ref.replace("Patient/", "")
            else:
                patient_fhir_id = None

            hadm_id = None
            for ident in rec.get("identifier", []):
                if ident.get("system") == "http://mimic.mit.edu/fhir/mimic/identifier/encounter-hosp":
                    hadm_id = ident.get("value")
                    break

            period = rec.get("period", {})
            start_time = period.get("start")
            end_time = period.get("end")

            if fhir_id and hadm_id and patient_fhir_id:
                records_buffer.append({
                    "encounter_fhir_id": fhir_id,
                    "hadm_id": hadm_id,
                    "patient_fhir_id": patient_fhir_id,
                    "start_time": start_time,
                    "end_time": end_time
                })
            records_processed += 1
            if len(records_buffer) >= batch_size:
                df_batch = pd.DataFrame(records_buffer)
                con.execute("INSERT INTO encounters SELECT * FROM df_batch")
                records_buffer = []
                gc.collect()

        if records_buffer:
            df_batch = pd.DataFrame(records_buffer)
            con.execute("INSERT INTO encounters SELECT * FROM df_batch")

    final_count = con.execute("SELECT COUNT(*) FROM encounters").fetchone()[0]
    logging.info("Loaded %d encounter records into DuckDB (unique inserted: %d).",
                 records_processed, final_count)

def process_condition_data(con, condition_file, sepsis_codes, pneumonia_codes):
    """
    Reads MIMIC condition data from NDJSON.gz, focuses on sepsis/pneumonia codes + a sample of others.
    Populates 'conditions' DuckDB table with is_sepsis, is_pneumonia flags.
    """
    logging.info("Processing condition data from %s", os.path.basename(condition_file))
    con.execute("""
        CREATE TABLE IF NOT EXISTS conditions (
            condition_fhir_id VARCHAR,
            encounter_fhir_id VARCHAR,
            patient_fhir_id VARCHAR,
            icd_code VARCHAR,
            icd_system VARCHAR,
            icd_display VARCHAR,
            is_sepsis BOOLEAN,
            is_pneumonia BOOLEAN
        )
    """)
    existing_count = con.execute("SELECT COUNT(*) FROM conditions").fetchone()[0]
    if existing_count > 0:
        logging.info("conditions table already has %d records, skipping reload.", existing_count)
        return

    sepsis_set = set([c.lower() for c in sepsis_codes])
    pneumonia_set = set([c.lower() for c in pneumonia_codes])
    records_buffer = []
    batch_size = 10000
    processed_count = 0

    if not os.path.exists(condition_file):
        logging.warning("Condition file not found: %s. Skipping.", condition_file)
        return

    with gzip.open(condition_file, 'rt', encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fhir_id = rec.get("id")
            enc_ref = rec.get("encounter", {}).get("reference", None)
            subj_ref = rec.get("subject", {}).get("reference", None)
            if enc_ref and enc_ref.startswith("Encounter/"):
                encounter_fhir_id = enc_ref.replace("Encounter/", "")
            else:
                encounter_fhir_id = None
            if subj_ref and subj_ref.startswith("Patient/"):
                patient_fhir_id = subj_ref.replace("Patient/", "")
            else:
                patient_fhir_id = None

            coding_list = rec.get("code", {}).get("coding", [])
            if not coding_list:
                continue

            # We only store the first code for simplicity
            code_obj = coding_list[0]
            icd_code = (code_obj.get("code") or "").lower()
            icd_system = code_obj.get("system", "")
            icd_display = code_obj.get("display", "")
            is_sepsis = icd_code in sepsis_set or "sepsis" in icd_display.lower()
            is_pneumonia = icd_code in pneumonia_set or "pneumonia" in icd_display.lower()

            # Keep or skip
            # We keep all sepsis/pneumonia conditions or ~10% sample of others
            keep_record = False
            if is_sepsis or is_pneumonia:
                keep_record = True
            else:
                # Sample
                if processed_count % 10 == 0:
                    keep_record = True

            if keep_record and fhir_id and encounter_fhir_id and patient_fhir_id:
                records_buffer.append({
                    "condition_fhir_id": fhir_id,
                    "encounter_fhir_id": encounter_fhir_id,
                    "patient_fhir_id": patient_fhir_id,
                    "icd_code": icd_code,
                    "icd_system": icd_system,
                    "icd_display": icd_display,
                    "is_sepsis": is_sepsis,
                    "is_pneumonia": is_pneumonia
                })

            processed_count += 1
            if len(records_buffer) >= batch_size:
                df_batch = pd.DataFrame(records_buffer)
                con.execute("INSERT INTO conditions SELECT * FROM df_batch")
                records_buffer = []
                gc.collect()

        # flush
        if records_buffer:
            df_batch = pd.DataFrame(records_buffer)
            con.execute("INSERT INTO conditions SELECT * FROM df_batch")

    final_count = con.execute("SELECT COUNT(*) FROM conditions").fetchone()[0]
    logging.info("Processed %d lines from conditions; stored %d rows in DuckDB.",
                 processed_count, final_count)

# ------------------------------------------------------------------------------
# Data Integration: Note Data
# ------------------------------------------------------------------------------
def process_discharge_notes(con, note_file, target_patients=None, chunk_size=5000):
    """
    Reads discharge notes in chunks from a CSV (compressed with gzip if needed).
    Loads only the subset of patients we care about (e.g., with sepsis or pneumonia).
    Cleans text if needed, or you can do separate text pipeline.

    Expects columns: subject_id, hadm_id, text, ...
    """
    logging.info("Processing discharge notes from %s", os.path.basename(note_file))
    con.execute("""
        CREATE TABLE IF NOT EXISTS discharge_notes (
            note_id VARCHAR,
            subject_id VARCHAR,
            hadm_id VARCHAR,
            note_type VARCHAR,
            note_seq INTEGER,
            charttime TIMESTAMP,
            storetime TIMESTAMP,
            text VARCHAR
        )
    """)
    existing_count = con.execute("SELECT COUNT(*) FROM discharge_notes").fetchone()[0]
    if existing_count > 0:
        logging.info("discharge_notes table already has %d records, skipping reload.", existing_count)
        return

    # If we want to filter to only target_patients
    if target_patients is not None:
        target_set = set(target_patients)
        logging.info("Found %d target patients for notes filtering.", len(target_set))
    else:
        target_set = None

    notes_inserted = 0
    rows_processed = 0

    if not os.path.exists(note_file):
        logging.warning("Discharge notes file not found: %s. Skipping.", note_file)
        return

    for chunk in pd.read_csv(note_file, chunksize=chunk_size, compression='infer'):
        chunk["subject_id"] = chunk["subject_id"].astype(str)
        # Filter if target_patients is used
        if target_set is not None:
            chunk = chunk[chunk["subject_id"].isin(target_set)]

        if not chunk.empty:
            # Basic text field truncation if needed
            # chunk["text"] = chunk["text"].apply(lambda x: x[:65000] if isinstance(x, str) else x)
            try:
                con.execute("INSERT INTO discharge_notes SELECT * FROM chunk")
                inserted_size = len(chunk)
            except Exception as e:
                logging.error("Batch insert error, trying row by row: %s", e)
                inserted_size = 0
                for _, row in chunk.iterrows():
                    try:
                        con.execute(
                            "INSERT INTO discharge_notes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            [
                                row.get("note_id"),
                                row.get("subject_id"),
                                row.get("hadm_id"),
                                row.get("note_type"),
                                row.get("note_seq"),
                                row.get("charttime"),
                                row.get("storetime"),
                                row.get("text")
                            ]
                        )
                        inserted_size += 1
                    except Exception as row_e:
                        logging.warning("Skipping single note insertion error: %s", row_e)

            notes_inserted += inserted_size
        rows_processed += len(chunk)

    logging.info("Processed %d rows, inserted %d notes into 'discharge_notes'.", rows_processed, notes_inserted)

# ------------------------------------------------------------------------------
# Text Cleaning Pipeline
# ------------------------------------------------------------------------------
def remove_boilerplate(text):
    """
    Removes known boilerplate headers/footers. Modify as needed.
    """
    if not isinstance(text, str):
        return text
    # Example: remove "HEADER:" lines
    text = re.sub(r"(?i)^header:.*$", "", text, flags=re.MULTILINE)
    # Remove trailing page numbers
    text = re.sub(r"Page \d+ of \d+", "", text)
    return text

def deidentify_text(text):
    """
    Removes or replaces certain personal identifiers or patterns.
    """
    if not isinstance(text, str):
        return text
    # Example: remove dates
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{2,4}", "[DATE]", text)
    # Remove potential names
    text = re.sub(r"\[.*?Name.*?\]", "[NAME]", text, flags=re.IGNORECASE)
    return text

def correct_ocr_errors(text):
    """
    Example OCR correction routine for typical misreads.
    """
    if not isinstance(text, str):
        return text
    # A small set of potential replacements:
    corrections = {
        "0f": "of",
        "vv": "w"
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

def clean_text_pipeline(text):
    """
    Runs a combined pipeline to remove boilerplate, de-ID, fix OCR, etc.
    """
    text = remove_boilerplate(text)
    text = deidentify_text(text)
    text = correct_ocr_errors(text)
    # Lowercase
    if isinstance(text, str):
        text = text.lower()
    return text

def preprocess_notes_in_db(con):
    """
    Applies text cleaning to the 'discharge_notes' table in DuckDB, updating the text column or creating a new column.
    This could be memory heavy if done on large data; consider chunk-based approach if needed.
    """
    logging.info("Preprocessing text in discharge_notes table.")
    # We'll do a simple approach: read in batches from DB, process in Python, write back
    # For large datasets, a chunk-based approach or ephemeral table might be needed.
    count_query = "SELECT COUNT(*) FROM discharge_notes"
    total_rows = con.execute(count_query).fetchone()[0]
    logging.info("Total notes: %d", total_rows)
    if total_rows == 0:
        return

    # Create a new column: cleaned_text
    con.execute("ALTER TABLE discharge_notes ADD COLUMN IF NOT EXISTS cleaned_text VARCHAR")

    offset = 0
    batch_size = 5000
    processed_rows = 0

    while True:
        query = f"""
            SELECT note_id, text 
            FROM discharge_notes 
            LIMIT {batch_size} OFFSET {offset}
        """
        df_batch = con.execute(query).df()
        if df_batch.empty:
            break
        # Clean text
        df_batch["cleaned_text"] = df_batch["text"].apply(clean_text_pipeline)

        # Update DB
        for _, row in df_batch.iterrows():
            update_query = """
                UPDATE discharge_notes 
                SET cleaned_text = ? 
                WHERE note_id = ?
            """
            con.execute(update_query, [row["cleaned_text"], row["note_id"]])
        processed_rows += len(df_batch)
        offset += batch_size
        logging.info("Processed %d / %d note rows", processed_rows, total_rows)
        if processed_rows >= total_rows:
            break

    logging.info("Completed text preprocessing for discharge_notes. Processed %d rows total.", processed_rows)

# ------------------------------------------------------------------------------
# Create Focused Views, Disease Progression, etc.
# ------------------------------------------------------------------------------
def create_focused_views(con):
    """
    Creates specialized DuckDB views for sepsis/pneumonia analysis.
    E.g., patient_basics, sepsis_conditions, pneumonia_conditions, etc.
    """
    logging.info("Creating targeted views in DuckDB.")
    con.execute("""
        CREATE OR REPLACE VIEW patient_basics AS
        SELECT
            p.subject_id,
            p.gender,
            p.birth_date,
            e.hadm_id,
            e.encounter_fhir_id,
            e.start_time,
            e.end_time
        FROM patients p
        JOIN encounters e ON p.fhir_id = e.patient_fhir_id
    """)
    con.execute("""
        CREATE OR REPLACE VIEW sepsis_conditions AS
        SELECT
            c.*,
            e.hadm_id,
            e.start_time AS encounter_time,
            p.subject_id,
            p.gender,
            p.birth_date
        FROM conditions c
        JOIN encounters e ON c.encounter_fhir_id = e.encounter_fhir_id
        JOIN patients p ON c.patient_fhir_id = p.fhir_id
        WHERE c.is_sepsis = TRUE
    """)
    con.execute("""
        CREATE OR REPLACE VIEW pneumonia_conditions AS
        SELECT
            c.*,
            e.hadm_id,
            e.start_time AS encounter_time,
            p.subject_id,
            p.gender,
            p.birth_date
        FROM conditions c
        JOIN encounters e ON c.encounter_fhir_id = e.encounter_fhir_id
        JOIN patients p ON c.patient_fhir_id = p.fhir_id
        WHERE c.is_pneumonia = TRUE
    """)
    # Additional combined or progression views, etc.
    logging.info("Views for sepsis and pneumonia created successfully.")

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    """
    Main entry point for data integration & preprocessing:
     1. Initialize DuckDB
     2. Process patients, encounters, conditions (focused on sepsis/pneumonia)
     3. Process discharge notes (optionally filtering only relevant patients)
     4. Preprocess text using cleaning pipeline
     5. Create or update views for downstream analysis
    """
    logging.info("=== Starting data integration & preprocessing script ===")

    # 1. Init DuckDB
    con = initialize_duckdb(DB_OUTPUT_PATH)

    # 2. Process core FHIR data
    # <<< Adjust file paths as needed
    patient_file = os.path.join(FHIR_DIR, "MimicPatient.ndjson.gz")
    encounter_file = os.path.join(FHIR_DIR, "MimicEncounter.ndjson.gz")
    condition_file = os.path.join(FHIR_DIR, "MimicCondition.ndjson.gz")

    process_patient_data(con, patient_file)
    process_encounter_data(con, encounter_file)

    # Example ICD codes for sepsis/pneumonia
    sepsis_codes = ["A419", "R6521", "99592", "0389"]  # <<< add your own
    pneumonia_codes = ["J189", "486", "J158", "4829"]  # <<< add your own
    process_condition_data(con, condition_file, sepsis_codes, pneumonia_codes)

    # 3. Process note data
    discharge_file = os.path.join(NOTES_DIR, "discharge.csv.gz")
    # Optional: If we only want patients with sepsis/pneumonia, gather them
    target_patients = con.execute("""
        SELECT DISTINCT p.subject_id
        FROM patients p
        JOIN encounters e ON p.fhir_id = e.patient_fhir_id
        JOIN conditions c ON e.encounter_fhir_id = c.encounter_fhir_id
        WHERE c.is_sepsis = TRUE OR c.is_pneumonia = TRUE
    """).df()["subject_id"].tolist()

    process_discharge_notes(con, discharge_file, target_patients=target_patients, chunk_size=5000)

    # 4. Preprocess text in DB
    preprocess_notes_in_db(con)

    # 5. Create specialized views
    create_focused_views(con)

    con.close()
    logging.info("=== Data integration & preprocessing script completed successfully ===")

if __name__ == "__main__":
    main()