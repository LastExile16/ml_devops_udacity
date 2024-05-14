import joblib
import os
import logging
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pandas as pd
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
    """
    Fixture to initialize ChurnLibrary object for testing.

    Returns:
        ChurnLibrary: Initialized ChurnLibrary object.
    """
    try:
        # Initialize ChurnLibrary object
        obj = ChurnLibrary("./data/bank_data.csv")
        logging.info("ChurnLibrary object initialized successfully.")
        
        # Generate synthetic data for testing
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        # Create a DataFrame with dummy column names
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_df = pd.DataFrame(y, columns=['target'])
        X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.5, random_state=42)
        
        # Attach data variables to the ChurnLibrary object
        obj.X_train = X_train
        obj.X_test = X_test
        obj.y_train = y_train
        obj.y_test = y_test
        obj.X = X_df  # Attach DataFrame with dummy column names
        
    except FileNotFoundError as err:
        logging.error("FileNotFoundError occurred during object initialization: %s", err)
        raise err
    
    # Return the ChurnLibrary object as the fixture
    return obj

def test_import(cl_object):
    """
    Test data import.

    Parameters:
        cl_object (ChurnLibrary): ChurnLibrary object to test.

    Raises:
        AssertionError: If the DataFrame shape is not as expected.
    """
    try:
        assert cl_object.df.shape[0] > 0
        assert cl_object.df.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err

def test_eda(cl_object):
    """
    Test perform_eda function.

    Parameters:
        cl_object (ChurnLibrary): ChurnLibrary object to test.

    Raises:
        AssertionError: If expected EDA images are not generated.
    """
    try:
        cl_object.perform_eda()
        logging.info("Test perform_eda: Function call SUCCESS")
    except Exception as err:
        logging.error(f"Test perform_eda: {err}")
        raise err
    try:
        images_dir = './images'
        # Verify that the images have been created in the specified directory
        assert os.path.exists(os.path.join(images_dir, "churn_vs_active_histogram.png")), "EDA churn_vs_active_histogram data not generated"
        assert os.path.exists(os.path.join(images_dir, "customer_age_histogram.png")), "EDA customer_age_histogram data not generated"
        assert os.path.exists(os.path.join(images_dir, "marital_status_barplot.png")), "EDA marital_status_barplot data not generated"
        assert os.path.exists(os.path.join(images_dir, "density_total_trans_count.png")), "EDA density_total_trans_count data not generated"
        assert os.path.exists(os.path.join(images_dir, "heatmap_corr.png")), "EDA heatmap_corr data not generated"
        logging.info(f"Test perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(f"Test perform_eda: {err}")
        raise err

def test_encoder_helper(cl_object):
    """
    Test encoder_helper function.

    Parameters:
        cl_object (ChurnLibrary): ChurnLibrary object to test.

    Raises:
        AssertionError: If expected new columns are not created or values are out of expected range.
    """
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    
    try:
        cl_object.encoder_helper(cat_columns, 'Churn')
        logging.info("Test encoder_helper: SUCCESS")
    except Exception as err:
        logging.error(f"Test encoder_helper: {err}")
        raise err
        
    # Verify the expected output
    expected_columns = [
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn'
    ]
    try:
        assert all(col in cl_object.df.columns for col in expected_columns), "New columns not created"

        # Assert that values in the new columns are within the expected range (0 to 1)
        for col in expected_columns:
            assert all(0 <= cl_object.df[col]) and all(cl_object.df[col] <= 1), f"Values in {col} column are not within the expected range"
        
    except AssertionError as err:
        logging.error(f"Testing encoder_helper: {err}")
        raise err
        
def test_perform_feature_engineering(cl_object):
    """
    Test perform_feature_engineering function.

    Parameters:
        cl_object (ChurnLibrary): ChurnLibrary object to test.

    Raises:
        AssertionError: If the shapes of the returned data do not match expectations.
    """
    # Call the function to be tested
    encoded_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category',
        'Attrition_Flag'
    ]
    cl_object.perform_feature_engineering('Churn', encoded_columns)
    logging.info(f"Test perform_feature_engineering: ....")
    # Verify the shapes of the returned data
    try: 
        assert cl_object.y_train.shape[1] == 1, "Number of response feature in y must be the 1"
        assert cl_object.X_train.shape[0] == cl_object.y_train.shape[0], "Number of samples in X_train and y_train must be the same"
        assert cl_object.X_test.shape[0] == cl_object.y_test.shape[0], "Number of samples in X_test and y_test must be the same"

        # Assert that train-test split sizes are correct (assuming test_size=0.3)
        assert abs(cl_object.X_train.shape[0] / cl_object.df.shape[0] - 0.7) < 0.01, "Incorrect train size"
        assert abs(cl_object.X_test.shape[0] / cl_object.df.shape[0] - 0.3) < 0.01, "Incorrect test size"
        logging.info(f"Test perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(f"Testing perform_feature_engineering: {err}")
        raise err

def test_classification_report_image(cl_object):
    """
    Test classification_report_image function.

    Parameters:
        cl_object (ChurnLibrary): ChurnLibrary object to test.

    Raises:
        AssertionError: If expected report images are not generated.
    """
    # Generate random predictions for logistic regression (0 or 1) with a threshold of 0.5
    y_train_preds_lr = np.random.choice([0, 1], size=len(cl_object.y_train), p=[0.5, 0.5])
    y_test_preds_lr = np.random.choice([0, 1], size=len(cl_object.y_test), p=[0.5, 0.5])

    # Generate random predictions for random forest (0 or 1) with a threshold of 0.5
    y_train_preds_rf = np.random.choice([0, 1], size=len(cl_object.y_train), p=[0.5, 0.5])
    y_test_preds_rf = np.random.choice([0, 1], size=len(cl_object.y_test), p=[0.5, 0.5])

    # Ensure the images folder exists
    images_dir = "./images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Call the classification_report_image method
    cl_object.classification_report_image(y_train_preds_lr, 
                                          y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)
    try:
        # Verify that the images have been created in the specified directory
        assert os.path.exists(os.path.join(images_dir, "rfc_classification_report.png")), "RFC report for test/train data not generated"
        assert os.path.exists(os.path.join(images_dir, "lrc_classification_report.png")), "LRC report for test/train data not generated"
        logging.info(f"Testing classification_report_image: SUCCESS")
    except AssertionError as err:
        logging.error(f"Testing classification_report_image: {err}")
        raise err

def test_plot_roc_curves(cl_object):
    """
    Test plot_roc_curves function.

    Parameters:
        cl_object (ChurnLibrary): ChurnLibrary object to test.

    Raises:
        AssertionError: If the ROC plot comparison for models is not generated.
    """
    # Grid search parameters
    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [4, 50, 100],
        'criterion': ['gini', 'entropy']
    }
    # Create mock models
    rfc = RandomForestClassifier(random_state=42)
    cl_object.cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cl_object.lrc = LogisticRegression(solver='lbfgs', random_state=42)

    # Train models (in real scenario, you would use cl_object to train models)
    cl_object.cv_rfc.fit(cl_object.X_train, cl_object.y_train)
    cl_object.lrc.fit(cl_object.X_train, cl_object.y_train)

    # Specify the save path for the ROC plot
    save_path = './images'

    # Call the plot_roc_curves method
    cl_object.roc_curves_comparison(save_path)
    try:
        # Assert that the saved image file exists
        assert os.path.exists(save_path), "ROC plot comparison for the models not generated"
        logging.info(f"Testing plot_roc_curves: SUCCESS")
    except AssertionError as err:
            logging.error(f"Testing plot_roc_curves: {err}")
            raise err

def test_scatter_plots_comparison(cl_object):
    """
    Test scatter_plots_comparison function.

    Parameters:
        cl_object (ChurnLibrary): ChurnLibrary object to test.

    Raises:
        AssertionError: If the scatter plot comparison for models is not generated.
    """
    # Grid search parameters
    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [4, 50, 100],
        'criterion': ['gini', 'entropy']
    }
    # Create mock models
    rfc = RandomForestClassifier(random_state=42)
    cl_object.cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cl_object.lrc = LogisticRegression(solver='lbfgs', random_state=42)

    # Train models (in real scenario, you would use cl_object to train models)
    cl_object.cv_rfc.fit(cl_object.X_train, cl_object.y_train)
    cl_object.lrc.fit(cl_object.X_train, cl_object.y_train)

    # Specify the save path for the scatterplot plot
    save_path = './images'

    cl_object.scatter_plots_comparison(save_path)
    try:
        # Assert that the saved image file exists
        assert os.path.exists(save_path), "Scatter plot comparison for the models not generated"
        logging.info(f"Testing scatter_plots_comparison: SUCCESS")
    except AssertionError as err:
            logging.error(f"Testing scatter_plots_comparison: {err}")
            raise err

def test_feature_importance_plot(cl_object):
    """
    Test feature_importance_plot function.

    Parameters:
        cl_object (ChurnLibrary): ChurnLibrary object to test.

    Raises:
        AssertionError: If the feature importance plots for the models are not generated.
    """
    # Grid search parameters
    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [4, 50, 100],
        'criterion': ['gini', 'entropy']
    }
    # Create mock models
    rfc = RandomForestClassifier(random_state=42)
    cl_object.cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cl_object.cv_rfc.fit(cl_object.X_train, cl_object.y_train)

    # Specify the save path for the ROC plot
    output_pth = './images'

    cl_object.feature_importance_plot(ChurnLibrary.TEST_XDATA, output_pth)
    try:
        # Assert that the saved image files exist
        assert os.path.exists(f"{output_pth}/feature_summaryplot_bar_multiclass.png"), "Feature summaryplot bar multiclass plot for the models not generated"
        assert os.path.exists(f"{output_pth}/feature_summaryplot_dot.png"), "Feature summaryplot dot plot for the models not generated"
        assert os.path.exists(f"{output_pth}/feature_gridsearch.png"), "Feature gridsearch plot for the models not generated"
        logging.info(f"Testing feature_importance_plot: SUCCESS")
    except AssertionError as err:
        logging.error(f"Testing feature_importance_plot: {err}")
        raise err
    
def test_train_models(cl_object):
    """
    Test train_models function.

    Parameters:
        cl_object (ChurnLibrary): ChurnLibrary object to test.

    Raises:
        AssertionError: If models are not trained and saved correctly.
    """
    # Call the function to be tested
    cl_object.train_models()

    try:
        logging.info(f"Test train_models: ...")
        assert hasattr(cl_object, 'cv_rfc'), "cv_rfc attribute is not set"
        assert hasattr(cl_object, 'lrc'), "lrc attribute is not set"
        assert hasattr(cl_object.cv_rfc, 'best_estimator_'), "cv_rfc best estimator not set"
        assert hasattr(cl_object.lrc, 'predict'), "lrc predict method not set"
        
        # Check if the models are saved
        rfc_model_path = os.path.join('./models/', 'rfc_model.pkl')
        assert os.path.exists(rfc_model_path), "Random Forest model not saved"
        lrc_model_path = os.path.join('./models/', 'logistic_model.pkl')
        assert os.path.exists(lrc_model_path), "Logistic Regression model not saved"
        
        # Load the saved models and ensure they are of the correct type
        rfc_model = joblib.load(rfc_model_path)
        assert isinstance(rfc_model, type(cl_object.cv_rfc.best_estimator_)), "Loaded Random Forest model type mismatch"
        lrc_model = joblib.load(lrc_model_path)
        assert isinstance(lrc_model, type(cl_object.lrc)), "Loaded Logistic Regression model type mismatch"
        logging.info(f"Test train_models: SUCCESS")
    except AssertionError as err:
        logging.error(f"Testing train_models: {err}")
        raise err
