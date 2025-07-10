import os
import sys
import torch
import pickle
import argparse
import warnings
import contextlib
from pathlib import Path
from tqdm.auto import tqdm
from esm.models.esm3 import ESM3
import torch.multiprocessing as mp
from esm.sdk.api import ESMProtein, GenerationConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL

# Suppress specific warnings to avoid clutter in the logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure PyTorch CUDA memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Define the directory of the current script
SCRIPT_DIR = Path(__file__).resolve().parent

# Define the path to store generated PDB files
DATA_PATH = SCRIPT_DIR / "data/esm/pdb"
DATA_PATH.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

# Add the parent directory to the system path for module imports
sys.path.append(str(SCRIPT_DIR.parent))

# Import the logger configuration
from logger_config import setup_logger

# Define the path for log files
LOG_PATH = SCRIPT_DIR / "logs/esm"
LOG_PATH.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

# Initialize the logger
logger = setup_logger(log_dir=LOG_PATH, log_file_name="esm3_pdbs.log")

@contextlib.contextmanager
def suppress_output():
    """
    Context manager to suppress stdout and stderr output.
    Useful for suppressing verbose library outputs.
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def predict_structures(gpu_id, proteins, use_cpu=False):
    """
    Generate PDB structures for a list of proteins using ESM3.

    Args:
        gpu_id (int): GPU ID to use for computation.
        proteins (list): List of protein dictionaries with 'sequence' and 'acc_id'.
        use_cpu (bool): Whether to use CPU instead of GPU. Defaults to False.
    """
    device = 'cpu' if use_cpu else f"cuda:{gpu_id}"  # Determine the device to use
    if not use_cpu:
        try:
            # Set the specified GPU as the current device
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()  # Clear the GPU memory cache
        except Exception as e:
            logger.error(f"Failed to set GPU {gpu_id} as the current device: {e}")
            raise e

    # Load the ESM3 model on the specified device
    client = ESM3.from_pretrained(ESM3_OPEN_SMALL).to(device)

    # Iterate over the list of proteins and generate PDB structures
    for protein in tqdm(proteins, position=gpu_id, desc=f"{f'CPU {gpu_id}' if use_cpu else f'GPU {gpu_id}'} - Generating PDBs", leave=False):
        esm_protein = ESMProtein(protein['sequence'])  # Create an ESMProtein object
        try:
            with suppress_output():
                # Generate the protein structure using the ESM3 model
                structure = client.generate(esm_protein, GenerationConfig(track="structure", num_steps=32))
            # Save the generated structure as a PDB file
            structure.to_pdb(DATA_PATH / f"ESM3-open-small-{protein['acc_id']}.pdb")
            logger.info(f"Successfully generated structure for {protein['acc_id']} on GPU {gpu_id}.")
        except:
            # Log a warning if structure generation fails
            logger.warning(f"Failed to generate structure for {protein['acc_id']} on GPU {gpu_id}.")
            torch.cuda.empty_cache()  # Clear the GPU memory cache in case of failure

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate PDBs from sequences using ESM3")
    parser.add_argument("--input_file", type=str, help="Path to the pickle file containing protein sequences", default=DATA_PATH.parent.parent / "uniprot_sequences.pkl")
    args = parser.parse_args()

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()

    # Load the protein sequences from the input pickle file
    with open(args.input_file, "rb") as f:
        sequences = pickle.load(f)
    logger.info(f"Number of proteins without PDB structures in AFDB: {len(sequences)}")

    # Filter sequences to those with length <= 2048 (ESM3 model constraint)
    sequences = [prot for prot in sequences if len(prot["sequence"]) <= 2048]
    logger.info(f"Filtered sequences to {len(sequences)} with length <= 2048.")

    # Further filter sequences to exclude those with existing PDB files
    sequences = [prot for prot in sequences if not (DATA_PATH / f"ESM3-open-small-{prot['acc_id']}.pdb").exists()]
    logger.info(f"Filtered sequences to {len(sequences)} that do not have a ESM3 generated PDB file already.")

    # Split the sequences into chunks for parallel processing across GPUs
    logger.info("Splitting sequences into chunks for each GPU...")
    proteins_chunks = [sequences[i::num_gpus] for i in range(num_gpus)]
    
    logger.info(f"Generating PDBs for {len(sequences)} proteins using {num_gpus} GPUs.")
    processes = []
    try:
        # Start a separate process for each GPU
        for gpu_id in range(num_gpus):
            p = mp.Process(target=predict_structures, args=(gpu_id, proteins_chunks[gpu_id]))
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        # Handle keyboard interruption and terminate all processes
        print("Interrupted! Terminating processes...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        print("All processes terminated.")