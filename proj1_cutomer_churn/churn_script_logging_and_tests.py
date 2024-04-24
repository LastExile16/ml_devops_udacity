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

def _test_import(cl_object):
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

def _test_eda(cl_object):
    '''
    test perform eda function
    '''
    try:
        cl_object.perform_eda()
        logging.info("Test *perform_eda*: SUCCESS")
    except Exception as err:
        logging.error(f"Test *perform_eda*: {err}")
        raise err
        
def test_encoder_helper(cl_object):
    '''
    test encoder helper
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    
    try:
        cl_object.encoder_helper(cat_columns, 'Churn')
        logging.info("Test *encoder_helper*: SUCCESS")
    except Exception as err:
        logging.error(f"Test *encoder_helper*: {err}")
        raise err
        
    # Verify the expected output
    expected_columns = [
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn'
    ]
    assert all(col in cl_object.df.columns for col in expected_columns), "New columns not created"

    # Assert that values in the new columns are within the expected range (0 to 1)
    for col in expected_columns:
        assert all(0 <= cl_object.df[col]) and all(cl_object.df[col] <= 1), f"Values in {col} column are not within the expected range"
        
        
        
def test_perform_feature_engineering(cl_object):

    # Call the function to be tested
    X_train, X_test, y_train, y_test = cl_object.perform_feature_engineering('Churn')

    # Verify the shapes of the returned data
    assert X_train.shape[0] == y_train.shape[0], "Number of samples in X_train and y_train must be the same"
    assert X_test.shape[0] == y_test.shape[0], "Number of samples in X_test and y_test must be the same"

    # Assert that train-test split sizes are correct (assuming test_size=0.3)
    assert abs(X_train.shape[0] / cl_object.df.shape[0] - 0.7) < 0.01, "Incorrect train size"
    assert abs(X_test.shape[0] / cl_object.df.shape[0] - 0.3) < 0.01, "Incorrect test size"
