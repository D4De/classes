import os
import logging
import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.conf')
logging.config.fileConfig(log_file_path)

def get_logger(name : str) -> logging.Logger:
    return logging.getLogger(name)