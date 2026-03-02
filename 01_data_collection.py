import logging
import pandas as pd
import requests
import os
import time

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


TARGET_CHEMBL_ID = 'CHEMBL203'
BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
OUTPUT_FILE = "data/egfr_bioactivity_data.csv"
BATCH_SIZE = 100

def get_total_count():
    """Fetches the total number of records available for the target."""
    params = {
        'target_chembl_id': TARGET_CHEMBL_ID,
        'standard_type': 'IC50',
        'limit': 1
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data['page_meta']['total_count']
    except Exception as e:
        logging.info(f"Error fetching metadata: {e}")
        return 3000

def load_existing_data():
    """Loads existing CSV if available to support resuming."""
    if os.path.exists(OUTPUT_FILE):
        try:
            df = pd.read_csv(OUTPUT_FILE)
            logging.info(f"Found existing file with {len(df)} records.")
            return df
        except Exception as e:
            logging.info(f"Error reading existing file: {e}. Starting fresh.")
    return pd.DataFrame()

def save_data(df):
    """Saves DataFrame to CSV."""
    if not os.path.exists("data"):
        os.makedirs("data")
    df.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Saved {len(df)} records to {OUTPUT_FILE}.")

def fetch_and_update():
    df_current = load_existing_data()
    if not df_current.empty:
        existing_ids = set(df_current['molecule_chembl_id'].tolist())
    else:
        existing_ids = set()

    total_count = get_total_count()
    logging.info(f"Target Total Records: {total_count}")


    for offset in range(0, total_count, BATCH_SIZE):
        params = {
            'target_chembl_id': TARGET_CHEMBL_ID,
            'standard_type': 'IC50',
            'limit': BATCH_SIZE,
            'offset': offset,
            'only': 'molecule_chembl_id,canonical_smiles,standard_value,standard_units,pchembl_value'
        }

        try:
            logging.info(f"Fetching batch offset {offset}/{total_count}...")
            response = requests.get(BASE_URL, params=params, timeout=15)

            if response.status_code != 200:
                logging.info(f"Error: Status {response.status_code}. Retrying...")
                time.sleep(2)
                response = requests.get(BASE_URL, params=params, timeout=15)
                if response.status_code != 200:
                    logging.info(f"Skipping batch {offset} due to error.")
                    continue

            data = response.json()
            activities = data.get('activities', [])

            if not activities:
                logging.info("Empty batch received.")
                continue

            df_batch = pd.DataFrame(activities)

            if 'standard_value' in df_batch.columns and 'canonical_smiles' in df_batch.columns:
                df_batch = df_batch.dropna(subset=['standard_value', 'canonical_smiles'])
            else:
                continue

            if df_current.empty:
                df_current = df_batch
                new_records_count = len(df_batch)
            else:
                df_current = pd.concat([df_current, df_batch], ignore_index=True)

                before_len = len(df_current)
                df_current = df_current.drop_duplicates(subset=['molecule_chembl_id', 'standard_value'])
                new_records_count = len(df_current) - before_len

            if new_records_count > 0:
                logging.info(f"Added {new_records_count} new records.")
                save_data(df_current)
            else:
                logging.info("No new unique records in this batch.")

            time.sleep(0.2)

        except Exception as e:
            logging.info(f"Exception at batch {offset}: {e}")
            continue

    return df_current

def clean_final_data(df):
    logging.info("\nPerforming final integrity check and cleaning...")

    if df.empty:
        logging.info("Dataset is empty.")
        return df

    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')

    df = df.dropna(subset=['standard_value', 'canonical_smiles'])

    df = df.drop_duplicates(subset=['canonical_smiles'])

    logging.info(f"Final valid unique compounds: {len(df)}")
    save_data(df)
    return df

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")

    logging.info("=== STARTING ROBUST DATA COLLECTION ===")
    df = fetch_and_update()

    if not df.empty:
        clean_final_data(df)
        logging.info("=== DATA COLLECTION COMPLETED ===")
    else:
        logging.info("=== FAILED: NO DATA COLLECTED ===")
        exit(1)
