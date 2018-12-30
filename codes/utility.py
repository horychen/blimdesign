
import os
import logging
import datetime

def myLogger(dir_codes, prefix='opti_script_'): # This works even when the module is reloaded (which is not the case of the other answers) https://stackoverflow.com/questions/7173033/duplicate-log-output-when-using-python-logging-module
    logger=logging.getLogger()
    if not len(logger.handlers):
        logger.setLevel(logging.DEBUG)
        now = datetime.datetime.now()

        # create a file handler
        handler=logging.FileHandler(dir_codes + prefix + now.strftime("%Y-%m-%d") +'.log')
        handler.setLevel(logging.DEBUG)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)
    return logger

def logger_init(): # This will lead to duplicated logging output
    # logger = logging.getLogger(__name__) # this is used in modules 
    logger = logging.getLogger() # use this (root) in the main executable file
    logger.setLevel(logging.DEBUG)

    # create a file handler
    now = datetime.datetime.now()
    handler = logging.FileHandler(dir_codes + r'opti_script.log')
    handler.setLevel(logging.DEBUG)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    return logger
