import os
import sys
import torch
import pickle
import argparse
import warnings
from pathlib import Path
from tqdm.auto import tqdm
from esm.models.esmc import ESMC
import torch.multiprocessing as mp
from esm.sdk.api import ESMProtein, LogitsConfig

# Suppress FutureWarnings to keep the output clean
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure PyTorch CUDA memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Define the script directory
SCRIPT_DIR = Path(__file__).resolve().parent

# Define the data directory
DATA_PATH = SCRIPT_DIR / "data"

# Define and create the AlphaFold data directory
AF_PATH = SCRIPT_DIR / "data/alphafold/pdb"
AF_PATH.mkdir(parents=True, exist_ok=True)

# Define and create the ESM data directory
ESM_PATH = SCRIPT_DIR / "data/esm/pdb"
ESM_PATH.mkdir(parents=True, exist_ok=True)

# Add the parent directory to the system path for imports
sys.path.append(str(SCRIPT_DIR.parent))

# Import the logger configuration
from logger_config import setup_logger

# Define and create the log directory
LOG_PATH = SCRIPT_DIR / "logs/esm"
LOG_PATH.mkdir(parents=True, exist_ok=True)

# Set up the logger
logger = setup_logger(log_dir=LOG_PATH, log_file_name="esmc_embeddings.log")

def get_embeddings(sequences, gpu_id):
    """
    Generate embeddings for a list of protein sequences using the ESMC model.

    Args:
        sequences (list): List of protein sequences.
        gpu_id (int): GPU ID to use for computation.

    Returns:
        dict: A dictionary mapping sequences to their embeddings.
    """
    # Set the GPU device for PyTorch
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()  # Clear the GPU cache

    # Load the pre-trained ESMC model onto the specified GPU
    client = ESMC.from_pretrained("esmc_600m").to(f"cuda:{gpu_id}")

    # Dictionary to store embeddings
    protein_embeddings = {}

    # Process each sequence
    for seq in tqdm(sequences, position=gpu_id, desc=f"GPU {gpu_id} - Processing embeddings", leave=False):
        try:
            # Create an ESMProtein object from the sequence
            protein = ESMProtein(seq)

            # Encode the protein sequence into a tensor
            protein_tensor = client.encode(protein)

            # Generate logits and extract embeddings
            logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))

            # Compute the mean of the embeddings and store it
            protein_embeddings[seq] = logits_output.embeddings.mean(dim=1).detach().cpu().numpy().squeeze()
        except:
            pass  # Ignore errors for problematic sequences

    return protein_embeddings

def process_embeddings(gpu_id, sequences_chunk, return_dict):
    """
    Process a chunk of sequences on a specific GPU and store the results in a shared dictionary.

    Args:
        gpu_id (int): GPU ID to use for computation.
        sequences_chunk (list): Chunk of sequences to process.
        return_dict (multiprocessing.Manager().dict): Shared dictionary to store results.
    """
    # Generate embeddings for the chunk of sequences
    embeddings = get_embeddings(sequences_chunk, gpu_id)

    # Store the embeddings in the shared dictionary
    return_dict[gpu_id] = embeddings

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate ESMC embeddings from protein sequences")
    parser.add_argument("--input_file", type=str, default=DATA_PATH / "uniprot_sequences.pkl", help="Path to the input file containing protein sequences")
    args = parser.parse_args()

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    logger.info(f"Number of GPUs available: {num_gpus}")
    
    # Load the protein sequences from the input file
    logger.info(f"Loading sequences from {args.input_file}")
    with open(args.input_file, 'rb') as f:
        sequences = pickle.load(f)
    logger.info(f"Loaded {len(sequences)} sequences from {args.input_file}")

    # Split the sequences into chunks for parallel processing
    logger.info(f"Splitting sequences into {num_gpus} chunks for multiprocessing")
    sequences_chunks = [sequences[i::num_gpus] for i in range(num_gpus)]

    # Create a multiprocessing manager and shared dictionary
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    try:
        # Start a separate process for each GPU
        for gpu_id in range(num_gpus):
            p = mp.Process(target=process_embeddings, args=(gpu_id, sequences_chunks[gpu_id], return_dict))
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Combine the results from all GPUs
        protein_embeddings = {}
        for gpu_id in range(num_gpus):
            protein_embeddings.update(return_dict[gpu_id])

        # Save the embeddings to a file
        output_file = DATA_PATH / (Path(args.input_file).stem + "_esmc_embeddings.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(protein_embeddings, f)
        logger.info(f"Sequence embeddings saved to {output_file}")

    except KeyboardInterrupt:
        # Handle keyboard interrupt and terminate processes
        logger.error("Interrupted! Terminating processes...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        logger.error("All processes terminated.")