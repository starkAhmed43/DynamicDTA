import sys
import json
import pickle
import argparse
from torch import cuda
from pathlib import Path
from unimol_tools import UniMolRepr
from MolR.featurizer import MolEFeaturizer

# Define the directory of the current script
SCRIPT_DIR = Path(__file__).resolve().parent 

# Define the data path and ensure the directory exists
DATA_PATH = SCRIPT_DIR / "data/"
DATA_PATH.mkdir(parents=True, exist_ok=True)

# Add the parent directory to the system path for module imports
sys.path.append(str(SCRIPT_DIR.parent))
from logger_config import setup_logger

# Define the log path and ensure the directory exists
LOG_PATH = SCRIPT_DIR / "logs/smiles"
LOG_PATH.mkdir(parents=True, exist_ok=True)

# Set up the logger for logging messages
logger = setup_logger(log_dir=LOG_PATH, log_file_name="embed_mol_graph.log")

def get_molr_2d_embeddings(input_file, output_file):
    """
    Generate 2D molecular embeddings for a list of SMILES strings using the MolR model.

    Args:
        input_file (str): Path to the input JSON file containing SMILES strings.
        output_file (str): Path to save the generated 2D embeddings as a pickle file.
    """
    try:
        # Initialize the MolR featurizer model
        model = MolEFeaturizer()
        logger.info("MolR model loaded successfully.")
    except Exception as e:
        logger.error(f"Error while loading MolR model: {e}")
        raise e

    try:
        # Load SMILES strings from the input JSON file
        with open(input_file, "r") as f:
            smiles = json.load(f)
        smiles = [smi["smiles"] for smi in smiles]  # Extract SMILES strings
        logger.info(f"Loaded {len(smiles)} SMILES strings from {input_file}.")
    except Exception as e:
        logger.error(f"Error while loading SMILES strings from {input_file}: {e}")
        raise e

    try:
        # Generate 2D embeddings for the SMILES strings
        logger.info("Generating 2D embeddings...")
        embeddings, flags = model.transform(smiles)
        mol_2d_embeddings = {smile: embedding for smile, embedding in zip(smiles, embeddings)}
        logger.info(f"Generated 2D embeddings for {len(mol_2d_embeddings)} SMILES strings.")
    except Exception as e:
        logger.error(f"Error while generating 2D embeddings: {e}")
        raise e
    
    try:
        # Save the generated 2D embeddings to the output file
        logger.info(f"Saving 2D embeddings to {output_file}...")
        with open(output_file, "wb") as f:
            pickle.dump(mol_2d_embeddings, f)
        logger.info(f"Saved 2D embeddings to {output_file}.")
    except Exception as e:
        logger.error(f"Error while saving 2D embeddings to {output_file}: {e}")
        raise e
    
def get_unimol_3d_embeddings(input_file, output_file):
    """
    Generate 3D molecular embeddings for a list of SMILES strings using the UniMol-Tools model.

    Args:
        input_file (str): Path to the input JSON file containing SMILES strings.
        output_file (str): Path to save the generated 3D embeddings as a pickle file.
    """
    try:
        # Check if CUDA is available and initialize the UniMol model accordingly
        if cuda.is_available():
            logger.info("CUDA is available. Using GPU for computations.")
            clf = UniMolRepr(data_type='molecule', remove_hs=False, use_cuda=True)
        else:
            logger.info("CUDA is not available. Using CPU for computations.")
            clf = UniMolRepr(data_type='molecule', remove_hs=False, use_cuda=False)
        logger.info("UniMol model loaded successfully.")
    except Exception as e:
        logger.error(f"Error while loading UniMol model: {e}")
        raise e

    try:
        # Load SMILES strings from the input JSON file
        with open(input_file, "r") as f:
            smiles = json.load(f)
        smiles = [smi["smiles"] for smi in smiles]  # Extract SMILES strings
        logger.info(f"Loaded {len(smiles)} SMILES strings from {input_file}.")
    except Exception as e:
        logger.error(f"Error while loading SMILES strings from {input_file}: {e}")
        raise e
    
    try:
        # Generate 3D embeddings for the SMILES strings
        logger.info("Generating 3D embeddings...")
        unimol_repr = clf.get_repr(smiles, return_atomic_reprs=False)
        mol_3d_embeddings = {smile: embedding for smile, embedding in zip(smiles, unimol_repr['cls_repr'])}
        logger.info(f"Generated 3D embeddings for {len(mol_3d_embeddings)} SMILES strings.")
    except Exception as e:
        logger.error(f"Error while generating 3D embeddings: {e}")
        raise e
    
    try:
        # Save the generated 3D embeddings to the output file
        logger.info(f"Saving 3D embeddings to {output_file}...")
        with open(output_file, "wb") as f:
            pickle.dump(mol_3d_embeddings, f)
        logger.info(f"Saved 3D embeddings to {output_file}.")
    except Exception as e:
        logger.error(f"Error while saving 3D embeddings to {output_file}: {e}")
        raise e
    
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate 2D and 3D embeddings for SMILES strings.")
    parser.add_argument("--input_file", type=str, default=DATA_PATH / 'substrates_pubchem_smiles.json', help="Path to the JSON file containing SMILES strings")
    
    args = parser.parse_args()

    # Validate that the input file is a JSON file
    input_file = Path(args.input_file)
    if input_file.suffix != ".json":
        logger.error(f"Input file {args.input_file} is not a JSON file.")
        raise ValueError(f"Input file {args.input_file} is not a JSON file.")
    
    logger.info(f"Input file: {input_file}")
    
    # Generate 2D and 3D embeddings and save them to respective files
    get_molr_2d_embeddings(input_file, DATA_PATH / (input_file.stem + "_2d_embeddings.pkl"))
    #get_unimol_3d_embeddings(input_file, DATA_PATH / (input_file.stem + "_3d_embeddings.pkl"))