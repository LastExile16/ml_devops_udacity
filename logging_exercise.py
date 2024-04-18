## STEPS TO COMPLETE ##
# 1. import logging
# 2. set up config file for logging called `results.log`
# 3. add try except with logging for success or error
#    in relation to checking the types of a and b
# 4. check to see that log is created and populated correctly
#    should have error for first function and success for
#    the second
# 5. use pylint and autopep8 to make changes
#    and receive 10/10 pep8 score

import logging

# logging.basicConfig(
#     filename='./results.log',
#     level=logging.ERROR,
#     filemode='a',
#     format='%(asctime)s: %(name)s, %(funcName)s, - %(levelname)s - %(message)s')

import os
import logging.config

import yaml

def setup_logging(
    default_path='logging.yaml',
    default_level=logging.INFO,
    env_key='LOG_CFG'
):
    """Setup logging configuration

    """
    path = default_path
    
    value = os.getenv(env_key, None)
    
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config_str = f.read()
            config_str = config_str.replace('__name__', __name__)  # Replace '__name__' with the actual module name
            config = yaml.safe_load(config_str)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
        
# # Run once at startup:
# with open("conf.yaml", 'rt') as f:
#     config = yaml.safe_load(f.read())
# logging.config.dictConfig(config)

# # Include in each module:
# log = logging.getLogger(__name__)
# log.debug("Logging is configured.")

def sum_vals(a, b):
    '''
    Args:
        a: (int)
        b: (int)
    Return:
        a + b (int)
    '''
    setup_logging()
    # Example usage of the 'my_module' logger
    my_module_logger = logging.getLogger(__name__)
    my_module_logger.error(f'This is an error message from {__name__}')

    # Example usage of the 'root' logger
    root_logger = logging.getLogger()
    root_logger.info('This is an info message from the root logger')
    result = None
    try:
        result = int(a)+int(b)
        logging.debug("successfully calculated the sum!")
    except ValueError as e:
        logging.error(e)
    return result

if __name__ == "__main__":
    sum_vals('no', 'way')
    sum_vals(4, 5)
    print(__name__)