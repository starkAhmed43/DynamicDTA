import sys
import json
import torch
import pickle
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, pipeline

# Define the directory of the current script
SCRIPT_DIR = Path(__file__).resolve().parent 

# Define the data path and ensure it exists
DATA_PATH = SCRIPT_DIR / "data/"
DATA_PATH.mkdir(parents=True, exist_ok=True)

# Add the parent directory to the system path for imports
sys.path.append(str(SCRIPT_DIR.parent))
from logger_config import setup_logger

# Define the log path and ensure it exists
LOG_PATH = SCRIPT_DIR / "logs/smiles"
LOG_PATH.mkdir(parents=True, exist_ok=True)

# Set up the logger for logging messages
logger = setup_logger(log_dir=LOG_PATH, log_file_name="embed_smiles.log")

# Load the BARTSmiles model and tokenizer
logger.info("Loading BARTSmiles model...")
try:
    # Load the tokenizer for the BARTSmiles model
    tokenizer = AutoTokenizer.from_pretrained("gayane/BARTSmiles", add_prefix_space=True)
    tokenizer.pad_token = '<pad>'  # Set the padding token explicitly

    # Load the BARTSmiles model
    model = AutoModel.from_pretrained('gayane/BARTSmiles')
    model.eval()  # Set the model to evaluation mode

    # Create a pipeline for feature extraction
    extractor = pipeline("feature-extraction", model=model, tokenizer=tokenizer, device="cuda" if torch.cuda.is_available() else "cpu")
except Exception as e:
    logger.error(f"Error while loading BARTSmiles model: {e}")
    raise e
logger.info(f"BARTSmiles model loaded successfully on {extractor.device} device.")

def tokens_less_than_128(smi):
    """
    Check if the number of tokens in the SMILES string is less than 128.

    Args:
        smi (str): A SMILES string.

    Returns:
        bool: True if the number of tokens is less than or equal to 128, False otherwise.
    """
    global tokenizer
    # Tokenize the SMILES string and check the number of tokens
    return tokenizer(smi, padding=True, truncation=False, return_tensors="pt", add_special_tokens=True)["input_ids"].shape[1] <= 128

def get_bartsmiles_embedding(smiles, batch_size=128, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Get the BARTSmiles embedding for a list of SMILES strings using AutoModel directly.

    Args:
        smiles (list): List of SMILES strings.
        batch_size (int): Batch size for processing SMILES strings.
        device (str): Device to use for computation ('cuda' or 'cpu').

    Returns:
        dict: A dictionary mapping SMILES strings to their embeddings.
    """
    global tokenizer, model

    # Move the model to the specified device
    model.to(device)

    # Filter out SMILES strings with token lengths > 128
    valid_smiles = [smiles[i] for i in tqdm(range(len(smiles)), desc="Filtering SMILES strings with >128 tokens") if tokens_less_than_128(smiles[i])]
    logger.info(f"Filtered out {len(smiles) - len(valid_smiles)} SMILES strings with token lengths > 128.")
    smiles = valid_smiles

    # Tokenize the SMILES strings
    logger.info("Tokenizing SMILES strings...")
    tokenized = tokenizer(smiles, padding=True, truncation=False, return_tensors="pt", add_special_tokens=True)
    if "token_type_ids" in tokenized:
        del tokenized["token_type_ids"]  # Remove token_type_ids if present (not needed for BART)

    # Move tokenized inputs to the specified device
    tokenized = {key: val.to(device) for key, val in tokenized.items()}

    # Initialize variables for storing embeddings and tracking failures
    smiles_embedding = {}
    failed_count = 0

    # Process valid SMILES strings in batches
    logger.info(f"Getting BARTSmiles embedding for {len(valid_smiles)} SMILES strings...")
    for i in tqdm(range(0, len(valid_smiles), batch_size), desc="Embedding SMILES batches"):
        batch_smiles = valid_smiles[i:i + batch_size]
        try:
            # Extract embeddings directly from the model
            with torch.no_grad():
                outputs = model(**{key: val[i:i + batch_size] for key, val in tokenized.items()})
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling over sequence length

            # Store embeddings in the dictionary
            for j, smile in enumerate(batch_smiles):
                smiles_embedding[smile] = batch_embeddings[j].cpu().numpy()
        except Exception as e:
            failed_count += len(batch_smiles)
            logger.error(f"Error while processing batch {i // batch_size + 1}: {e}")
    
    logger.info(f"Successfully embedded {len(smiles_embedding)} SMILES strings.")
    logger.info(f"Failed to embed {failed_count} SMILES strings.")
    return smiles_embedding

def embed_smiles():
    """
    Embed the SMILES strings using BARTSmiles and save the embeddings to a file.

    This function reads SMILES strings from a JSON file, generates embeddings using the BARTSmiles model,
    and saves the embeddings to a pickle file.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate 1D embeddings from SMILES strings")
    parser.add_argument("--input_file", type=str, help="Path to the json file containing SMILES strings", default=DATA_PATH / 'substrates_pubchem_smiles.json')
    args = parser.parse_args()

    smiles_input_file = Path(args.input_file)

    # Check if the input file exists
    if not smiles_input_file.exists():
        logger.error(f"Input file {smiles_input_file} does not exist.")
        raise FileNotFoundError(f"Input file {smiles_input_file} does not exist.")
    
    # Check if the input file is a JSON file
    if smiles_input_file.suffix != ".json":
        logger.error(f"Input file {smiles_input_file} is not a json file.")
        raise ValueError(f"Input file {smiles_input_file} is not a json file.")

    logger.info(f"Input file: {smiles_input_file}")
    
    # Define the output file path
    smiles_output_file = DATA_PATH / (smiles_input_file.stem + "_1d_embeddings.pkl")
    logger.info(f"Output file: {smiles_output_file}")

    try:
        # Load SMILES strings from the input JSON file
        with open(smiles_input_file, "r") as f:
            smiles = json.load(f)
        smiles = [smi["smiles"] for smi in smiles]  # Extract SMILES strings from the JSON data
        logger.info(f"SMILES strings loaded from {smiles_input_file}")
    except Exception as e:
        logger.error(f"Error while loading SMILES strings: {e}")
        raise e
    
    # Generate embeddings for the SMILES strings
    smiles_embedding = get_bartsmiles_embedding(smiles)
    logger.info("SMILES strings embedded successfully.")
    
    try:
        # Save the embeddings to the output file
        with open(smiles_output_file, "wb") as f:
            pickle.dump(smiles_embedding, f)
        logger.info(f"SMILES embeddings saved to {smiles_output_file}")
    except Exception as e:
        logger.error(f"Error while saving SMILES embeddings: {e}")
        raise e

    logger.info("SMILES embedding process completed successfully.")

if __name__ == "__main__":
    embed_smiles()