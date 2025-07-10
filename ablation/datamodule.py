import sys
import torch
import numpy as np
from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split

# Define paths for the script directory and data directory
SCRIPT_DIR = Path(__file__).resolve().parent 
DATA_PATH = SCRIPT_DIR.parent / "data"

# Add the parent directory to the system path for importing modules
sys.path.append(str(SCRIPT_DIR.parent))
from logger_config import setup_logger

# Define the path for logs and ensure the directory exists
LOG_PATH = SCRIPT_DIR.parent / "logs/training"
LOG_PATH.mkdir(parents=True, exist_ok=True)

# Set up the logger for logging information
logger = setup_logger(log_dir=LOG_PATH, log_file_name="datamodule.log")

class DTADataset(Dataset):
    """
    A PyTorch Dataset class for handling DTA (Drug-Target Affinity) data and molecular embeddings.
    
    Attributes:
        rmsf (torch.Tensor): Tensor of root mean square fluctuation values.
        gyr (torch.Tensor): Tensor of radius of gyration values.
        div_se (torch.Tensor): Tensor of sequence diversity values.
        div_mm (torch.Tensor): Tensor of molecular diversity values.
        affinity (torch.Tensor): Tensor of affinity values.
        smiles_embed (torch.Tensor): Tensor of 1D molecular embeddings.
        mol_2d_embed (torch.Tensor): Tensor of 2D molecular embeddings.
        mol_3d_embed (torch.Tensor): Tensor of 3D molecular embeddings.
        esmc_embed (torch.Tensor): Tensor of ESMC embeddings.
        esm3_embed (torch.Tensor): Tensor of ESM3 embeddings.
    """
    def __init__(
            self,
            rmsf, gyr, div_se, div_mm, affinity,
            smiles_embed, mol_2d_embed, mol_3d_embed,
            esmc_embed, esm3_embed):
        """
        Initialize the dataset with various molecular and affinity data.

        Args:
            rmsf (torch.Tensor): Root mean square fluctuation values.
            gyr (torch.Tensor): Radius of gyration values.
            div_se (torch.Tensor): Sequence diversity values.
            div_mm (torch.Tensor): Molecular diversity values.
            affinity (torch.Tensor): Affinity values.
            smiles_embed (torch.Tensor): 1D molecular embeddings.
            mol_2d_embed (torch.Tensor): 2D molecular embeddings.
            mol_3d_embed (torch.Tensor): 3D molecular embeddings.
            esmc_embed (torch.Tensor): ESMC embeddings.
            esm3_embed (torch.Tensor): ESM3 embeddings.
        """
        super().__init__()

        self.rmsf = rmsf
        self.gyr = gyr
        self.div_se = div_se
        self.div_mm = div_mm
        self.affinity = affinity
        self.smiles_embed = smiles_embed
        self.mol_2d_embed = mol_2d_embed
        self.mol_3d_embed = mol_3d_embed
        self.esmc_embed = esmc_embed
        self.esm3_embed = esm3_embed

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.affinity)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the combined embedding and various molecular properties.
        """

        # Concatenate all embeddings into a single tensor
        embedding = torch.cat((
            self.smiles_embed[idx],
            self.mol_2d_embed[idx],
            self.mol_3d_embed[idx],
            self.esmc_embed[idx],
            self.esm3_embed[idx],
            self.rmsf[idx].unsqueeze(0),
            self.gyr[idx].unsqueeze(0),
            self.div_se[idx].unsqueeze(0),
            self.div_mm[idx].unsqueeze(0),
        ), dim=0)

        # Log the index and shape of the embedding
        logger.debug(f"Index: {idx}")
        logger.debug(f"Embedding shape: {embedding.shape}")
        
        return (
            embedding,
            self.affinity[idx].unsqueeze(0)
        )
    
class DTADataModule(LightningDataModule):

    def __init__(
            self,
            train_df,
            test_df,
            num_workers=4,
            batch_size=32,
    ):
        """
        Initialize the DataModule with the required parameters.

        Args:
            train_file_name (str): Name of the training data file containing DTA data.
            test_file_name (str): Name of the test data file containing DTA data.
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.
            batch_size (int, optional): Batch size for data loaders. Defaults to 32.
        """
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.num_workers = num_workers
        self.batch_size = batch_size

        logger.info(f"Initializing DTADataModule with training DataFrame, "
                    f"test DataFrame, num_workers: {self.num_workers}, "
                    f"batch_size: {self.batch_size}")
        
    def parse_df(self, data):
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Data columns: {data.columns.tolist()}")

        affinity = torch.tensor(data['affinity'].values, dtype=torch.float32)
        rmsf = torch.tensor(data['Avg.RMSF'].values, dtype=torch.float32)
        gyr = torch.tensor(data['Avg.gyr'].values, dtype=torch.float32)
        div_se = torch.tensor(data['Div.SE'].values, dtype=torch.float32)
        div_mm = torch.tensor(data['Div.MM'].values, dtype=torch.float32)
        smiles_embed = torch.tensor(np.stack(data['smiles_1d_embedding'].values), dtype=torch.float32)
        mol_2d_embed = torch.tensor(np.stack(data['smiles_2d_embedding'].values), dtype=torch.float32)
        mol_3d_embed = torch.tensor(np.stack(data['smiles_3d_embedding'].values), dtype=torch.float32)
        esmc_embed = torch.tensor(np.stack(data['esmc_embedding'].values), dtype=torch.float32)
        esm3_embed = torch.tensor(np.stack(data['esm3_embedding'].values), dtype=torch.float32)

        logger.info(f"Affinity shape: {affinity.shape}")
        logger.info(f"RMSF shape: {rmsf.shape}")
        logger.info(f"Gyr shape: {gyr.shape}")
        logger.info(f"Div SE shape: {div_se.shape}")
        logger.info(f"Div MM shape: {div_mm.shape}")
        logger.info(f"SMILES embedding shape: {smiles_embed.shape}")
        logger.info(f"2D molecular embedding shape: {mol_2d_embed.shape}")
        logger.info(f"3D molecular embedding shape: {mol_3d_embed.shape}")
        logger.info(f"ESMC embedding shape: {esmc_embed.shape}")
        logger.info(f"ESM3 embedding shape: {esm3_embed.shape}")

        dataset = DTADataset(
            affinity, rmsf, gyr, div_se, div_mm,
            smiles_embed, mol_2d_embed, mol_3d_embed,
            esmc_embed, esm3_embed
        )
        logger.info(f"Dataset created with {len(dataset)} samples")

        return dataset

    def setup(self, stage=None):
        """
        Prepare the data by loading, preprocessing, and splitting it into train, validation, and test sets.

        Args:
            stage (str, optional): Stage of the setup (e.g., 'fit', 'test'). Defaults to None.
        """

        # Load the training dataset
        full_train_dataset = self.parse_df(self.train_df)
        logger.info(f"Full training dataset loaded with {len(full_train_dataset)} samples")

        # Random split into train and val (80%/20%)
        val_size = int(0.2 * len(full_train_dataset))
        train_size = len(full_train_dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset, [train_size, val_size]
        )
        logger.info(f"Training split: {len(self.train_dataset)} samples, Validation split: {len(self.val_dataset)} samples")

        # Load the test dataset
        self.test_dataset = self.parse_df(self.test_df)
        logger.info(f"Test dataset loaded with {len(self.test_dataset)} samples")

    def get_input_dim(self):
        """
        Get the input dimension of the dataset.

        Returns:
            int: Input dimension of the dataset.
        """
        # Ensure the datasets are initialized
        if not hasattr(self, "train_dataset") or self.train_dataset is None:
            logger.info("Calling setup method to initialize datasets...")
            self.setup()

        # Get the input dimension from the first sample in the training dataset
        return self.train_dataset[0][0].shape[0]


    def train_dataloader(self):
        """
        Create the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        """
        Create the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """
        Create the DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader for the test dataset.
        """
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )