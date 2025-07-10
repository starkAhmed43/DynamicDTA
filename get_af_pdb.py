import sys
import json
import time
import pickle
import argparse
import requests
from pathlib import Path
from tqdm.auto import tqdm
from multiprocessing.pool import ThreadPool

# Define the script directory
SCRIPT_DIR = Path(__file__).resolve().parent 

# Define the data path for AlphaFold-related files
DATA_PATH = SCRIPT_DIR / "data/alphafold"
DATA_PATH.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

# Define paths for JSON and PDB files
JSON_PATH = DATA_PATH / "json"
JSON_PATH.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

PDB_PATH = DATA_PATH / "pdb"
PDB_PATH.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

# Add the parent directory to the system path for module imports
sys.path.append(str(SCRIPT_DIR.parent))

# Import custom logger and database connection setup
from logger_config import setup_logger
from sqlite_connect import init_connection

# Define the log path for AlphaFold-related logs
LOG_PATH = SCRIPT_DIR / "logs/alphafold"
LOG_PATH.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

# Set up the logger
logger = setup_logger(log_dir=LOG_PATH, log_file_name="download_af_pdb.log")

def get_html(url, path, verbose=True):
    """
    Download the HTML content from a given URL and save it to the specified path.

    Args:
        url (str): The URL to fetch the content from.
        path (Path): The file path to save the content.
        verbose (bool): Whether to log messages or not.
    """
    if path.exists():  # Skip download if the file already exists
        return
    for _ in range(5):  # Retry up to 5 times in case of failure
        try:
            response = requests.get(url)
            if response.status_code == 200:  # Successful response
                with open(path, "w", encoding="utf-8") as f:
                    f.write(response.text)  # Save the content to the file
                return
            elif response.status_code == 404:  # Resource not found
                if verbose: 
                    logger.error(f"404 - {url}")
                return
        except Exception as e:  # Handle exceptions during the request
            if verbose: 
                logger.warning(f"Failed to retrieve {url} with error {e}. Retrying...")
            time.sleep(5)  # Wait before retrying
    else: 
        if verbose: 
            logger.error(f"Failed to retrieve {url}. Max retries exceeded.")
    return

def get_af_pdb(args):
    """
    Download PDB files from AlphaFold for a list of protein accession IDs.

    Args:
        args (list): List of protein accession IDs.
    """
    def get_af_pdb_helper(id):
        """
        Helper function to download JSON metadata and PDB files for a single protein ID.

        Args:
            id (str): Protein accession ID.
        """
        # Step 1: Download the JSON metadata for the protein
        json_url = f'https://alphafold.ebi.ac.uk/api/prediction/{id}'
        get_html(json_url, JSON_PATH / f"{id}.json")

        # Check if the JSON file was successfully downloaded
        if not (JSON_PATH / f"{id}.json").exists():
            return
        
        # Step 2: Parse the JSON file to get the PDB URL and download the PDB file
        with open(JSON_PATH / f"{id}.json", 'r') as f:
            af_cif_url = json.load(f)[0]["pdbUrl"]  # Extract the PDB URL
            af_cif_name = af_cif_url.split("/")[-1]  # Extract the PDB file name
        get_html(af_cif_url, PDB_PATH / af_cif_name)  # Download the PDB file

    logger.info("Downloading PDB files from AlphaFold...")
    # Use a thread pool to download files concurrently
    with tqdm(total=len(args), desc="Downloading PDB files from AlphaFold") as pbar:
        for _ in ThreadPool(20).imap_unordered(get_af_pdb_helper, args):
            pbar.update()  # Update the progress bar
    logger.info("PDB files downloaded successfully.")

def get_sequences_from_db(cursor, no_af_pdbs):
    """
    Query the protein_sequence table to find sequences for proteins without PDB structures.

    Args:
        cursor (sqlite3.Cursor): Database cursor for executing queries.
        no_af_pdbs (list): List of protein accession IDs without PDB structures.

    Returns:
        list: List of dictionaries containing accession IDs and sequences.
    """
    # Create a query with placeholders for the protein IDs
    placeholders = ', '.join(['?'] * len(no_af_pdbs))
    query = f"""
        SELECT acc_id, sequence
        FROM protein_sequence
        WHERE acc_id IN ({placeholders})
    """

    # Execute the query and fetch results
    cursor.execute(query, no_af_pdbs)
    results = [{"acc_id": row[0], "sequence": row[1]} for row in cursor.fetchall()]
    return results

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download PDB files from AlphaFold")
    parser.add_argument("--input_file", type=str, help="Path to the pickle file containing protein acc_ids", default=DATA_PATH.parent / "uniprot_ids.pkl")
    args = parser.parse_args()

    # Load protein accession IDs from the input file
    logger.info(f"Loading protein accession IDs from {args.input_file}...")
    with open(args.input_file, "rb") as f:
        acc_ids = pickle.load(f)

    # Download PDB files for the accession IDs
    get_af_pdb(acc_ids)

    # Log the number of proteins and those with/without PDB structures
    logger.info(f"Number of proteins: {len(acc_ids)}")
    af_pdbs = [acc for acc in acc_ids if (DATA_PATH / f"pdb/AF-{acc}-F1-model_v4.pdb").exists()]
    logger.info(f"Number of proteins with PDB structures in AFDB: {len(af_pdbs)}")

    logger.info(f"Number of proteins without PDB structures in AFDB: {len(acc_ids) - len(af_pdbs)}")
    no_af_pdbs = [acc for acc in acc_ids if not (DATA_PATH / f"pdb/AF-{acc}-F1-model_v4.pdb").exists()]
    
    # Connect to the database and extract sequences for proteins without PDB structures
    logger.info(f"Connecting to cache database to extract sequences for these proteins...")
    conn, cursor = init_connection()
    no_pdb_sequences = get_sequences_from_db(cursor, no_af_pdbs)
    logger.info(f"Loaded {len(no_pdb_sequences)} sequences from the database.")

    # Filter sequences to those with length <= 2048
    logger.info("Filtering sequences to those with length <= 2048.")
    no_pdb_sequences = [prot for prot in no_pdb_sequences if len(prot["sequence"]) <= 2048]
    logger.info(f"Filtered sequences to {len(no_pdb_sequences)} with length <= 2048.")

    # Save the filtered sequences to a pickle file
    logger.info(f"Saving sequences to {DATA_PATH / 'no_af_pdbs_sequences.pkl'} for ESM3 PDB generation.")
    with open(DATA_PATH / "no_af_pdbs_sequences.pkl", "wb") as f:
        pickle.dump(no_pdb_sequences, f)
    logger.info(f"Saved sequences to {DATA_PATH / 'no_af_pdbs_sequences.pkl'}.")