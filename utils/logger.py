import logging
import os
import sys
from datetime import datetime

class SimpleLogger:
    # Class variable to ensure all loggers use the same timestamp
    _shared_timestamp = None
    _base_log_dir = None
    
    def __init__(self, log_dir='logs', name=None, use_shared_timestamp=True, log_file=None):
        """
        Initialize a logger with shared timestamp to avoid multiple folders.
        
        Args:
            log_dir: Base directory to store log files
            name: Logger name. If None, timestamp will be used
            use_shared_timestamp: If True, all loggers share the same timestamp folder
            log_file: Optional specific log file path (overrides auto-generation)
        """
        if use_shared_timestamp:
            # Use shared timestamp for all loggers in the same experiment
            if SimpleLogger._shared_timestamp is None:
                SimpleLogger._shared_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
                SimpleLogger._base_log_dir = os.path.join(log_dir, SimpleLogger._shared_timestamp)
            
            self.log_dir = SimpleLogger._base_log_dir
        else:
            # Create individual timestamp (for separate experiments)
            if not os.path.isabs(log_dir):
                # If relative path, create timestamped subdirectory
                datetime_now = datetime.now().strftime('%Y-%m-%d_%H-%M')
                self.log_dir = os.path.join(log_dir, datetime_now)
            else:
                # If absolute path, use as-is
                self.log_dir = log_dir
        
        self.name = name if name else 'exp'
        
        # Get logger with unique name to avoid conflicts
        logger_name = f'{self.name}_{id(self)}'
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # Remove any existing handlers to avoid duplicates
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(console_formatter)
        self.logger.addHandler(ch)
        
        # File handler
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Use provided log_file or generate one
        if log_file:
            # If absolute path, use as-is; if relative, make it relative to log_dir
            if os.path.isabs(log_file):
                final_log_file = log_file
            else:
                final_log_file = os.path.join(self.log_dir, log_file)
        else:
            final_log_file = os.path.join(self.log_dir, f'{self.name}.log')
        
        # Ensure log file directory exists
        os.makedirs(os.path.dirname(final_log_file), exist_ok=True)
        
        fh = logging.FileHandler(final_log_file, mode='a')
        fh.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        fh.setFormatter(file_formatter)
        self.logger.addHandler(fh)
        
        if use_shared_timestamp and not hasattr(SimpleLogger, '_first_log_printed'):
            self.info(f"Shared experiment log directory: {self.log_dir}")
            SimpleLogger._first_log_printed = True

    @classmethod
    def reset_shared_timestamp(cls):
        """Reset shared timestamp for new experiment."""
        cls._shared_timestamp = None
        cls._base_log_dir = None
        if hasattr(cls, '_first_log_printed'):
            delattr(cls, '_first_log_printed')

    def info(self, message: str, *args, **kwargs) -> None:
        """Log info level message."""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning level message."""
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """Log error level message."""
        self.logger.error(message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug level message."""
        self.logger.debug(message, *args, **kwargs)