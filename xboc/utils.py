"""Define utilities."""

import logging

from rich.logging import RichHandler

FORMAT: str = "%(message)s"
DATE_FORMAT: str = "[%X]"


def setup_logging(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """Creates the logger for the BOCModel.

    Parameters
    ----------
    name : str
        Name for the logger
    log_level : int, optional
        Log level, by default logging.INFO

    Returns
    -------
    logging.Logger
        Logger with a rich handler
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    handler = RichHandler()
    handler.setFormatter(logging.Formatter(FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(handler)
    return logger
