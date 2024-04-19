import os
import logging
import pytest
from churn_library_class import ChurnLibrary

# Set up logging
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

@pytest.fixture
def cl_object():
    try:
        obj = ChurnLibrary("./data/bank_data.csv")
        logging.info("ChurnLibrary object initialized successfully.")
        return obj
    except FileNotFoundError as err:
        logging.error("FileNotFoundError occurred during object initialization: %s", err)
        raise err

def test_import(cl_object):
    '''
    Test data import.
    '''
    try:
        assert cl_object.df.shape[0] > 0
        assert cl_object.df.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err

def test_eda(cl_object):
    '''
    test perform eda function
    '''
    try:
        cl_object.perform_eda()
        logging.info("Test *perform_eda*: SUCCESS")
    except Exception as err:
        logging.error(f"Test *perform_eda*: {err}")
        raise err