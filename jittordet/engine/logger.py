"""Modified from mmcv.utils.logging."""
import logging

import jittor as jt

__all__ = ['get_logger', 'print_log']
initialized_loggers = set()


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    if name in initialized_loggers:
        return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if jt.rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if jt.rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    initialized_loggers.add(name)

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')
