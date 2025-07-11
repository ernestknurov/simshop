import logging 

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get a logger with the specified name and level.

    Args:
        name (str): The name of the logger.
        level (str): The logging level as string (default: "INFO").
                    Valid values: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

    Returns:
        logging.Logger: The logger instance.
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(numeric_level)
    return logger