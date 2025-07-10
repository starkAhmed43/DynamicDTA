import logging
from pathlib import Path
from tqdm.auto import tqdm

# Custom TqdmLoggingHandler for console output
class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler to integrate with tqdm.

    This handler ensures that log messages do not interfere with tqdm's progress bar
    by using tqdm's `write` method to output log messages.
    """
    def emit(self, record):
        """
        Emit a log record.

        Args:
            record (logging.LogRecord): The log record to be emitted.
        """
        try:
            # Format the log message
            msg = self.format(record)
            # Use tqdm.write to safely write the log message without breaking the progress bar
            tqdm.write(msg)
        except Exception:
            # Handle any errors that occur during logging
            self.handleError(record)

def setup_logger(log_dir, log_file_name):
    """
    Sets up a shared logger with customizable log directory and file name.

    This function configures a logger that writes log messages to both a file
    and the console (integrated with tqdm).

    Args:
        log_dir (str): Directory where the log file will be stored.
        log_file_name (str): Name of the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Resolve the absolute path of the log directory
    log_path = Path(log_dir).resolve()
    # Create the log directory if it doesn't already exist
    log_path.mkdir(parents=True, exist_ok=True)
    # Construct the full path to the log file
    log_file = log_path / log_file_name

    # Define the log message format
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Create a file handler to write log messages to the log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)  # Set the logging level for the file handler
    file_handler.setFormatter(logging.Formatter(log_format))  # Set the log message format

    # Create a console handler using the custom TqdmLoggingHandler
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(logging.INFO)  # Set the logging level for the console handler
    console_handler.setFormatter(logging.Formatter(log_format))  # Set the log message format

    # Get or create a logger instance named "shared_logger"
    logger = logging.getLogger("shared_logger")
    logger.setLevel(logging.INFO)  # Set the logging level for the logger
    logger.handlers = []  # Clear any existing handlers to avoid duplicate logs
    logger.addHandler(file_handler)  # Add the file handler to the logger
    logger.addHandler(console_handler)  # Add the console handler to the logger

    # Return the configured logger instance
    return logger