import logging
import os
import sys
from datetime import datetime

class SimpleLogger:
    def __init__(self, log_dir=None, name=None):
        """
        Initialize a basic logger.
        
        Args:
            log_dir: Directory to store log files. If None, no file logging will be used.
            name: Logger name. If None, timestamp will be used
        """
        self.name = name or datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Get logger with unique name
        self.logger = logging.getLogger(f'logger_{self.name}')
        self.logger.setLevel(logging.INFO)
        
        # Remove any existing handlers to avoid duplicates
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
        ch.setFormatter(console_formatter)
        self.logger.addHandler(ch)
        
        # Optional file handler
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f'{self.name}.log')
            fh = logging.FileHandler(log_file, mode='a')
            fh.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(file_formatter)
            self.logger.addHandler(fh)
            self.info(f"Log file: {log_file}")

    def info(self, message: str, *args, **kwargs) -> None:
        """Log info level message."""
        self.logger.info(message, *args, **kwargs)

