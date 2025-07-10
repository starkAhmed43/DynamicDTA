import os
import sys
import wandb
import argparse
from utils import *
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
from model import DTANN
from datamodule import DTADataModule
from pytorch_lightning.loggers import WandbLogger

# Set a random seed for reproducibility
pl.seed_everything(42, workers=True)

# Define the script directory
SCRIPT_DIR = Path(__file__).resolve().parent

# Add the parent directory to the system path for module imports
sys.path.append(str(SCRIPT_DIR.parent))

# Import the logger configuration
from logger_config import setup_logger

# Define the log directory and ensure it exists
LOG_PATH = SCRIPT_DIR.parent / "logs/training"
LOG_PATH.mkdir(parents=True, exist_ok=True)

# Set up the logger
logger = setup_logger(log_dir=LOG_PATH, log_file_name="hparam-search.log")

def train_model(config=None):
    """
    Train the model using the given configuration.

    Args:
        config (dict): Configuration dictionary containing hyperparameters.
    """
    # Initialize a Weights and Biases (wandb) run
    with wandb.init(config=config):
        config = wandb.config  # Access the configuration from wandb

    evaluations = {
        "s1": get_kfold_s1_splits,
        "s2": get_kfold_s2_splits,
        "s3": get_kfold_s3_splits,
        "s4": get_kfold_s4_splits,
    }

    dataset_df = pd.read_parquet(config.dataset_path)  # Load the dataset from a Parquet file
    
    k_fold_splits = evaluations[config.evaluation_type](dataset_df)  # Get the K-Fold splits based on the evaluation type

    for i, (train_indices, test_indices) in enumerate(k_fold_splits):
        logger.info(f"Evaluation:{config.evaluation_type} - Fold {i + 1}/{len(k_fold_splits)}")

        data_module = DTADataModule(
            train_df=dataset_df.iloc[train_indices],
            test_df=dataset_df.iloc[test_indices],
            num_workers=config.num_workers,
            batch_size=config.batch_size
        )

        logger.debug(f"Data Module: {data_module}")

        # Dynamically determine the input dimension from the data module
        input_dim = data_module.get_input_dim()

        # Initialize the model with the input dimension and hyperparameters
        model = DTANN(
            input_dim=input_dim,
            num_hid_layers=config.num_hid_layers,
            learning_rate=config.learning_rate,
        )
        logger.debug(f"Model: {model}")

        # Set up the Wandb logger for experiment tracking
        wandb_logger = WandbLogger(
            project=f"dynamic_dta_pred_{config.dataset}_{config.evaluation_type}",
            name=f"run_HL:{config.num_hid_layers}_LR:{config.learning_rate}_B:{config.batch_size}",
        )
        logger.debug(f"Wandb Logger: {wandb_logger}")

        # Initialize the PyTorch Lightning trainer
        trainer = pl.Trainer(
            max_epochs=config.max_epochs,  # Maximum number of training epochs
            accelerator="auto",  # Automatically select the appropriate accelerator (CPU/GPU)
            devices="auto",  # Automatically select the number of devices
            logger=wandb_logger,  # Use the Wandb logger
            enable_checkpointing=True,  # Enable checkpointing
            callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_rmse")],  # Monitor validation MSE for checkpointing
            precision="16-mixed",  # Use mixed precision for faster training
            deterministic=True,  # Ensure deterministic behavior for reproducibility
        )
        logger.debug(f"Trainer: {trainer}")

        # Start the training process
        trainer.fit(model, data_module)
        logger.info("Training completed.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Hyperparameter search for DTA Prediction")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        choices=["ic50", "ki", "kd"],
        help="Dataset to use for training (options: ic50, ki, kd)",
    )
    parser.add_argument(
        "--dataset_path", "-dp",
        type=str,
        required=True,
        help="Path to the dataset file",
    )
    parser.add_argument(
        "--evaluation_type", "-et",
        type=str,
        required=True,
        choices=["s1", "s2", "s3", "s4"],
        help="Type of evaluation to perform (options: s1, s2, s3, s4)",
    )
    parser.add_argument(
        "--sweep_count",
        type=int,
        default=50,
        help="Number of sweeps to run for hyperparameter optimization",
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_workers", "-nw",
        type=int,
        default=os.cpu_count(),
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--max_epochs", "-me",
        type=int,
        default=100,
        help="Maximum number of epochs for training",
    )
    args = parser.parse_args()

    # Validate the file name format
    if not args.dataset_path.endswith(".parquet"):
        logger.error("Invalid dataset file name. Ensure it ends with .parquet.")
        sys.exit(1)

    if not Path(args.dataset_path).exists():
        logger.error(f"Dataset path {args.dataset_path} does not exist.")
        sys.exit(1)    

    # Define the sweep configuration for hyperparameter optimization
    sweep_config = {
        "method": "bayes",  # Use Bayesian optimization for hyperparameter search
        "name": f"dynamic_dta_pred_{args.dataset}_{args.evaluation_type}",
        "metric": {
            "goal": "minimize",  # Minimize the validation mean squared error (MSE)
            "name": "val_mse",
        },
        "parameters": {
            "num_hid_layers": {
                "values": list(range(2, 12)),  # Number of hidden layers to search
            },
            "learning_rate": {
                "values": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],  # Learning rates to search
            },
            "batch_size": {
                "values": [args.batch_size],  # Fixed batch size
            },
            "num_workers": {
                "values": [args.num_workers],  # Fixed number of workers
            },
            "max_epochs": {
                "values": [args.max_epochs],  # Fixed maximum epochs
            },
            "dataset": {
                "values": [args.dataset],  # Fixed dataset
            },
            "dataset_path": {
                "values": [args.dataset_path],  # Fixed dataset path
            },
            "evaluation_type": {
                "values": [args.evaluation_type],  # Fixed evaluation type
            },
        }
    }

    # Initialize the sweep in Wandb
    sweep_id = wandb.sweep(
        sweep_config,
        project=f"dynamic_dta_pred_{args.dataset}_{args.evaluation_type}",  # Project name based on the dataset
    )
    logger.info(f"Sweep ID: {sweep_id}")
    
    # Start the sweep with the specified number of runs
    logger.info(f"Starting the sweep for {args.sweep_count} runs.")
    wandb.agent(sweep_id, train_model, count=args.sweep_count)
    logger.info("Sweep completed.")