"""
Script for util functions
"""
import os


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)

def make_list(x):
    return x if isinstance(x, list) else [x]

def list_to_string(values):
    return "_".join(str(x) for x in make_list(values))

import sys
import logging
import logging.config

def setup_logger(experiment_dir):
    def exception_hook(exc_type, exc_value, exc_traceback):
        logging.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

    default_logging = {
        'version': 1,
        'disable_existing_loggers': False,
        'propagate': False,
    }

    # Set up logging
    logging.config.dictConfig(default_logging)

    # Log file formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s():%(lineno)s] %(message)s",
        "%d/%m/%Y %H:%M:%S")

    # set a format which is simpler for console_log use
    formatter_slim = logging.Formatter(
        "[%(asctime)s] [%(name)s] %(message)s",
        "%H:%M:%S")

    # Setup logfile
    file_log = logging.FileHandler(os.path.join(experiment_dir, 'logFile.log'), mode='a')
    file_log.setLevel(logging.DEBUG)
    file_log.setFormatter(formatter)

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console_log = logging.StreamHandler(sys.stdout)
    lev = logging.INFO # if cfg.conf['debug'] > 1 else logging.INFO

    console_log.setLevel(lev)
    console_log.setFormatter(formatter_slim)  # tell the handler to use this format

    # Root logger
    root = logging.getLogger()
    root.handlers = []

    # add the handlers to the root logger
    root.addHandler(file_log)
    root.addHandler(console_log)
    root.setLevel(logging.DEBUG)

    # Also copy uncaught exceptions to log file
    sys.excepthook = exception_hook
