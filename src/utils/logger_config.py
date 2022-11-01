import sys
import os
import logging
from logging.config import dictConfig


class LogConfig:
    """
    HSD Mining Logging Configuration
    """
    def __init__(self):
        self.levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        self.logs_path = os.path.join(os.getcwd(), 'src', 'logs')

    def get_log_config(self):
        """Get logging configuration object

        Returns
        ----------
        dict: Logging configuration object
        """
        log_config = dict(
            version=1,
            formatters={
                'verbose': {
                    'format': ("[%(asctime)s] %(levelname)s "
                               "[%(name)s:%(lineno)s] %(message)s"),
                    'datefmt': "%m/%d/%Y %I:%M:%S %p",
                },
                'simple': {
                    'format': '%(levelname)s: %(message)s',
                },
            },
            handlers={
                'api-logger': {'class': 'logging.handlers.RotatingFileHandler',
                               'formatter': 'verbose',
                               'level': logging.INFO,
                               'filename': os.path.join(self.logs_path, 'api.log'),
                               'maxBytes': 52428800,
                               'backupCount': 7},
                'batch-process-logger': {'class': 'logging.handlers.RotatingFileHandler',
                                         'formatter': 'verbose',
                                         'level': logging.INFO,
                                         'filename': os.path.join(self.logs_path, 'batch.log'),
                                         'maxBytes': 52428800,
                                         'backupCount': 7},
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'simple',
                    'stream': sys.stdout,
                },
            },
            loggers={
                'api_logger': {
                    'handlers': ['api-logger', 'console'],
                    'level': logging.INFO
                },
                'batch_process_logger': {
                    'handlers': ['batch-process-logger', 'console'],
                    'level': logging.INFO
                }
            }
        )
        return log_config


config = LogConfig()
dictConfig(config.get_log_config())
api_logger = logging.getLogger('api_logger')
