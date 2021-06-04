"""
Script for util functions
"""
import os
import logging
import sys
import logging.config
from shutil import copytree, rmtree
import fnmatch


logger = logging.getLogger(__name__)

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)

def make_list(x):
    return x if isinstance(x, list) else [x]

def list_to_string(values):
    return "_".join(str(x) for x in make_list(values))

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

def include_patterns(*patterns):
    """
    Function that can be used as shutil.copytree() ignore parameter that
    determines which files *not* to ignore, the inverse of "normal" usage.

    This is a factory function that creates a function which can be used as a
    callable for copytree()'s ignore argument, *not* ignoring files that match
    any of the glob-style patterns provided.

    ‛patterns’ are a sequence of pattern strings used to identify the files to
    include when copying the directory tree.

    Example usage:

        copytree(src_directory, dst_directory,
                 ignore=include_patterns('*.sldasm', '*.sldprt'))
    """
    def _ignore_patterns(path, all_names):
        # Determine names which match one or more patterns (that shouldn't be
        # ignored).s
        keep = (name for pattern in patterns
                for name in fnmatch.filter(all_names, pattern))
        # # Ignore file names which *didn't* match any of the patterns given that
        # # aren't directory names.
        # dir_names = (name for name in all_names if isdir(join(path, name)))
        return set(all_names) - set(keep)

    return _ignore_patterns

def copy_source_code(exp_folder):
    """ Makes a copy of the Python scripts """

    src = sys.path[0]
    dst = os.path.join(exp_folder, 'Source_code')

    # Make sure the destination defaul_folder does not exist.
    if os.path.exists(dst) and os.path.isdir(dst):
        logger.info('Removing existing directory "{}"'.format(dst))
        rmtree(dst, ignore_errors=False)

    logger.info('Copying source code to "{}"'.format(dst))
    copytree(src, dst, ignore=include_patterns('quaternion', '*.py', '*.ipynb'))
