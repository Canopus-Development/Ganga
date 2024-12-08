
import logging
from pathlib import Path
from typing import Union

def setup_logging(level: str = "INFO", log_file: Union[str, Path] = None) -> logging.Logger:
    """Configure logging with file and console handlers"""
    # Create logger
    logger = logging.getLogger("ganga")
    logger.setLevel(level)
    
    # Create formatters
    fmt = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    
    # File handler if log_file specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    
    return logger