import colorlog
from datetime import datetime
import logging
import os
from typing import Union

log: Union[logging.Logger, None] = None


def configure_logging(name: str, verbose: bool) -> logging.Logger:
    global log
    log_colors = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    }

    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        log_colors=log_colors,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG if verbose else logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    log.addHandler(handler)

    log_filename = f"logs/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    log.addHandler(file_handler)

    return log
