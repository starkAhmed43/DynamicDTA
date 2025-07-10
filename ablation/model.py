import sys
import torch
import torch.nn as nn
from pathlib import Path
import torchmetrics as tm
import pytorch_lightning as pl

# Define the directory of the current script
SCRIPT_DIR = Path(__file__).resolve().parent

# Add the parent directory to the system path for importing custom modules
sys.path.append(str(SCRIPT_DIR.parent))
from logger_config import setup_logger

# Define the path for logging
LOG_PATH = SCRIPT_DIR.parent / "logs/training"
LOG_PATH.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

# Initialize the logger
logger = setup_logger(log_dir=LOG_PATH, log_file_name="model.log")

class DTANN(pl.LightningModule):
    """
    A PyTorch Lightning module for modeling kinetic laws using neural networks.
    This model is designed for regression tasks and supports dynamic layer configuration.

    Attributes:
        model (nn.Sequential): The feedforward neural network model.
        criterion (nn.MSELoss): The loss function for regression (Mean Squared Error).
        train_R2, val_R2, test_R2 (tm.R2Score): R2 score metrics for training, validation, and testing.
        train_pearson_r, val_pearson_r, test_pearson_r (tm.PearsonCorrCoef): Pearson correlation metrics.
        train_mae, val_mae, test_mae (tm.MeanAbsoluteError): Mean Absolute Error metrics.
        train_mse, val_mse, test_mse (tm.MeanSquaredError): Mean Squared Error metrics.
    """

    def __init__(self, input_dim, num_hid_layers, learning_rate=1e-3):
        """
        Initializes the KineticLawNN model with the specified input size, number of hidden layers, and learning rate.

        Args:
            input_dim (int): The size of the input features.
            num_hid_layers (int): The number of hidden layers in the model.
            learning_rate (float): The learning rate for the optimizer.
        """
        super(DTANN, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpointing and logging

        # Define the sizes of each layer in the network
        layer_sizes = [input_dim] + [2**i for i in range(num_hid_layers, 0, -1)] + [1]  # Output size is 1 for regression

        # Build the model architecture
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))  # Add a linear layer
            if i < len(layer_sizes) - 2:  # Add activation function for all layers except the last one
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)  # Combine layers into a sequential model

        # Define the loss function
        self.criterion = nn.MSELoss()

        # Define metrics for training, validation, and testing
        self.train_R2 = tm.R2Score()
        self.val_R2 = tm.R2Score()
        self.test_R2 = tm.R2Score()

        self.train_pearson_r = tm.PearsonCorrCoef()
        self.val_pearson_r = tm.PearsonCorrCoef()
        self.test_pearson_r = tm.PearsonCorrCoef()

        self.train_mae = tm.MeanAbsoluteError()
        self.val_mae = tm.MeanAbsoluteError()
        self.test_mae = tm.MeanAbsoluteError()

        self.train_rmse = tm.MeanSquaredError(squared=False)
        self.val_rmse = tm.MeanSquaredError(squared=False)
        self.test_rmse = tm.MeanSquaredError(squared=False)

        # Log model initialization details
        logger.info(f"DTANN model initialized with layer_sizes={layer_sizes}, learning_rate={learning_rate}")

    def on_train_start(self):
        """
        Called at the start of training. Resets training metrics.
        """
        self.train_R2.reset()
        self.train_pearson_r.reset()
        self.train_mae.reset()
        self.train_rmse.reset()

    def on_validation_start(self):
        """
        Called at the start of validation. Resets validation metrics.
        """
        self.val_R2.reset()
        self.val_pearson_r.reset()
        self.val_mae.reset()
        self.val_rmse.reset()

    def on_test_start(self):
        """
        Called at the start of testing. Resets test metrics.
        """
        self.test_R2.reset()
        self.test_pearson_r.reset()
        self.test_mae.reset()
        self.test_rmse.reset()

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        logger.debug(f"Forward pass input shape: {x.shape}")
        x = self.model(x)  # Pass input through the model
        logger.debug(f"Forward pass output shape: {x.shape}")
        return x

    def model_step(self, batch):
        """
        Performs a single forward pass and computes the loss for a given batch.

        Args:
            batch (tuple): A tuple containing input features and target values.

        Returns:
            tuple: Loss, predictions, and targets.
        """
        x, y = batch  # Unpack the batch
        preds = self.forward(x)  # Get predictions
        # Ensure no NaN or Inf values in predictions or targets
        assert not torch.isnan(preds).any(), "Predictions contain NaN values"
        assert not torch.isinf(preds).any(), "Predictions contain Inf values"
        assert not torch.isnan(y).any(), "Targets contain NaN values"
        assert not torch.isinf(y).any(), "Targets contain Inf values"
        loss = self.criterion(preds, y)  # Compute loss
        assert not torch.isnan(loss).any(), "Loss contains NaN values"
        logger.debug(f"Model step loss: {loss.item()}")
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        """
        Handles the training logic for a single batch.

        Args:
            batch (tuple): A tuple containing input features and target values.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        loss, preds, targets = self.model_step(batch)

        # Update training metrics
        logger.debug(f"Updating training metrics for batch {batch_idx}")
        self.train_R2(preds, targets)
        self.train_pearson_r(preds, targets)
        self.train_mae(preds, targets)
        self.train_rmse(preds, targets)

        # Log training metrics
        logger.debug(f"Logging training metrics for batch {batch_idx}")
        self.log("train_R2", self.train_R2, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_pearson_r", self.train_pearson_r, on_step=False, on_epoch=True)
        self.log("train_mae", self.train_mae, on_step=False, on_epoch=True)
        self.log("train_rmse", self.train_rmse, on_step=False, on_epoch=True, prog_bar=True)

        logger.debug(f"Training step {batch_idx}, MSELoss: {loss.item()}")
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Handles the validation logic for a single batch.

        Args:
            batch (tuple): A tuple containing input features and target values.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        loss, preds, targets = self.model_step(batch)

        # Update validation metrics
        logger.debug(f"Updating validation metrics for batch {batch_idx}")
        self.val_R2(preds, targets)
        self.val_pearson_r(preds, targets)
        self.val_mae(preds, targets)
        self.val_rmse(preds, targets)

        # Log validation metrics
        logger.debug(f"Logging validation metrics for batch {batch_idx}")
        self.log("val_R2", self.val_R2, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_pearson_r", self.val_pearson_r, on_step=False, on_epoch=True)
        self.log("val_mae", self.val_mae, on_step=False, on_epoch=True)
        self.log("val_rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=True)

        logger.debug(f"Validation step {batch_idx}, MSELoss: {loss.item()}")
        return loss

    def test_step(self, batch, batch_idx):
        """
        Handles the testing logic for a single batch.

        Args:
            batch (tuple): A tuple containing input features and target values.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        loss, preds, targets = self.model_step(batch)

        # Update test metrics
        logger.debug(f"Updating test metrics for batch {batch_idx}")
        self.test_R2(preds, targets)
        self.test_pearson_r(preds, targets)
        self.test_mae(preds, targets)
        self.test_rmse(preds, targets)

        # Log test metrics
        logger.debug(f"Logging test metrics for batch {batch_idx}")
        self.log("test_R2", self.test_R2, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_pearson_r", self.test_pearson_r, on_step=False, on_epoch=True)
        self.log("test_mae", self.test_mae, on_step=False, on_epoch=True)
        self.log("test_rmse", self.test_rmse, on_step=False, on_epoch=True, prog_bar=True)

        logger.debug(f"Test step {batch_idx}, MSELoss: {loss.item()}")
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            dict: A dictionary containing the optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)  # Use Adam optimizer
        logger.info(f"Adam optimizer configured with learning rate: {self.hparams.learning_rate}")
        return {'optimizer': optimizer}