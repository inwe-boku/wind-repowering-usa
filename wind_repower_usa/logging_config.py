import sys
import logging.config

from wind_repower_usa.config import LOG_FILE


setup_done = False


def log_exception(type_, value, traceback):
    logging.error("Uncaught exception:", exc_info=(type_, value, traceback))


def setup_logging(fname=LOG_FILE):
    global setup_done
    if setup_done:
        raise RuntimeError("Called setup_logging() twice, don't do this!")
    setup_done = True

    sys.excepthook = log_exception

    NO_COLOR = "\33[m"
    RED, GREEN, ORANGE, BLUE, PURPLE, LBLUE, GREY = (
        map("\33[%dm".__mod__, range(31, 38)))

    logging_config = {
        'version':  1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '[%(asctime)s] %(levelname)-4s - '
                          '%(name)-4s - %(message)s'
            },
            'color': {
                'format': '{}[%(asctime)s]{} {}%(levelname)-5s{} - '
                          '{}%(name)-5s{}: %(message)s'.format(
                              GREEN, NO_COLOR, PURPLE, NO_COLOR,
                              ORANGE, NO_COLOR)
            }
        },
        'handlers': {
            'stream': {
                'class': 'logging.StreamHandler',
                'formatter': 'color',
            }
        },
        'root': {
            'handlers': ['stream'],
            'level': logging.INFO,
        },
    }
    if fname is not None:
        logging_config['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'formatter': 'default',
            'level': logging.DEBUG,
            'filename': fname,
        }
        logging_config['root']['handlers'].append('file')

    logging.config.dictConfig(logging_config)

    logging.info("Starting %s....", sys.argv[0])
