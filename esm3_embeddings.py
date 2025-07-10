import os
import sys
import torch
import pickle
import argparse
import warnings
from pathlib import Path
from tqdm.auto import tqdm
from esm.models.esm3 import ESM3
import torch.multiprocessing as mp
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL

# Suppress FutureWarnings to keep the output clean
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure PyTorch CUDA memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Define the script directory
SCRIPT_DIR = Path(__file__).resolve().parent

# Define the data directory
DATA_PATH = SCRIPT_DIR / "data/"

# Define and create AlphaFold PDB directory
AF_PATH = DATA_PATH / "alphafold/pdb"
AF_PATH.mkdir(parents=True, exist_ok=True)

# Define and create ESM PDB directory
ESM_PATH = DATA_PATH / "esm/pdb"
ESM_PATH.mkdir(parents=True, exist_ok=True)

# Add the parent directory to the system path for imports
sys.path.append(str(SCRIPT_DIR.parent))

# Import custom logger configuration
from logger_config import setup_logger

# Define and create log directory
LOG_PATH = SCRIPT_DIR / "logs/esm"
LOG_PATH.mkdir(parents=True, exist_ok=True)

# Set up the logger
logger = setup_logger(log_dir=LOG_PATH, log_file_name="esm3_embeddings.log")

def get_embeddings(acc_ids, gpu_id):
    """
    Generate embeddings for a list of protein accession IDs using ESM3 model.

    Args:
        acc_ids (list): List of protein accession IDs.
        gpu_id (int): GPU ID to use for processing.

    Returns:
        dict: Dictionary mapping accession IDs to their embeddings.
    """
    # Set the GPU device and clear CUDA cache
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()

    # Load the ESM3 model on the specified GPU
    client = ESM3.from_pretrained(ESM3_OPEN_SMALL).to(f"cuda:{gpu_id}")
    protein_embeddings = {}

    # Iterate over each accession ID
    for acc in tqdm(acc_ids, position=gpu_id, desc=f"GPU {gpu_id} - Processing embeddings"):
        # Define paths for AlphaFold and ESM PDB files
        af_pdb_path = AF_PATH / f"AF-{acc}-F1-model_v4.pdb"
        esm_pdb_path = ESM_PATH / f"ESM3-open-small-{acc}.pdb"

        # Use AlphaFold PDB if it exists, otherwise use ESM PDB
        pdb_path = af_pdb_path if af_pdb_path.exists() else esm_pdb_path

        # Skip if no PDB file is found
        if not pdb_path.exists():
            logger.warning(f"Skipping {acc}: PDB file not found in either location.")
            continue

        try:
            # Log the start of processing
            logger.info(f"Processing {acc} on GPU {gpu_id}")

            # Load the protein structure from the PDB file
            protein = ESMProtein.from_pdb(pdb_path)

            # Encode the protein structure into a tensor
            protein_tensor = client.encode(protein)

            # Generate embeddings using the model
            output = client.forward_and_sample(
                protein_tensor,
                SamplingConfig(return_per_residue_embeddings=False, return_mean_embedding=True)
            )

            # Store the mean embedding in the dictionary
            protein_embeddings[acc] = output.mean_embedding.cpu().numpy()

            # Log successful processing
            logger.info(f"Successfully processed {acc} on GPU {gpu_id}")
        except Exception as e:
            # Log any errors encountered during processing
            logger.error(f"Error processing {acc} on GPU {gpu_id}: {e}")
            pass

    return protein_embeddings

def process_embeddings(gpu_id, acc_ids_chunk, return_dict):
    """
    Process a chunk of accession IDs on a specific GPU and store results in a shared dictionary.

    Args:
        gpu_id (int): GPU ID to use for processing.
        acc_ids_chunk (list): Chunk of accession IDs to process.
        return_dict (multiprocessing.Manager().dict): Shared dictionary to store results.
    """
    # Generate embeddings for the chunk of accession IDs
    embeddings = get_embeddings(acc_ids_chunk, gpu_id)

    # Store the embeddings in the shared dictionary
    return_dict[gpu_id] = embeddings

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate ESM3 embeddings from PDBs")
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the input file containing protein IDs",
        default=DATA_PATH / "uniprot_ids.pkl"
    )
    args = parser.parse_args()

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    logger.info(f"Number of GPUs available: {num_gpus}")

    # Load protein accession IDs from the input file
    logger.info(f"Loading protein IDs from {args.input_file}")
    with open(args.input_file, "rb") as f:
        acc_ids = pickle.load(f)
    logger.info(f"Loaded {len(acc_ids)} protein IDs")

    # Split the accession IDs into chunks for multiprocessing
    logger.info("Splitting protein IDs for multiprocessing")
    acc_ids_chunks = [acc_ids[i::num_gpus] for i in range(num_gpus)]

    # Create a multiprocessing manager and shared dictionary
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    try:
        # Start a process for each GPU
        for gpu_id in range(num_gpus):
            p = mp.Process(target=process_embeddings, args=(gpu_id, acc_ids_chunks[gpu_id], return_dict))
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Combine results from all GPUs
        protein_embeddings = {}
        for gpu_id in range(num_gpus):
            protein_embeddings.update(return_dict[gpu_id])

        # Save the embeddings to a file
        output_file = args.input_file.replace(".pkl", "_esm3_small_embeddings.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(protein_embeddings, f)
        logger.info(f"Embeddings saved to {output_file}")

    except KeyboardInterrupt:
        # Handle keyboard interrupt and terminate processes
        logger.error("Interrupted! Terminating processes...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        logger.error("All processes terminated.")