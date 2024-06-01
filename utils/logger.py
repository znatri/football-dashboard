import colorlog
from datetime import datetime
import logging
import os
import psutil
import torch

def configure_logging(name: str, verbose: bool) -> logging.Logger:
    log_colors = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
    
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        log_colors=log_colors,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    log_filename = f"logs/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)
    
    return logger

def log_resource_usage(log):
    process = psutil.Process(os.getpid())
    log.info(f"CPU usage: {process.cpu_percent()}%")
    log.info(f"Memory usage: {process.memory_info().rss / (1024 * 1024)} MB")
    if torch.cuda.is_available():
        log.info(f"Allocated GPU memory: {torch.cuda.memory_allocated() / (1024 * 1024)} MB")
        log.info(f"Cached GPU memory: {torch.cuda.memory_reserved() / (1024 * 1024)} MB")
