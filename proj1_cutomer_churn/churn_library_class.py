import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import pandas as pd

class ChurnLibrary:
    def __init__(self, pth):
        self.df = pd.read_csv(pth, header=0)

    def perform_eda(self):
        '''
        perform eda on df and save figures to images folder
        input:
                None

        output:
                None
        '''
        # Perform EDA operations on self.df and save figures

    def encoder_helper(self, category_lst, response):
        '''
        helper function to turn each categorical column into a new column with
        proportion of churn for each category - associated with cell 15 from the notebook

        input:
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could be used for naming variables or index y column]

        output:
                df: pandas dataframe with new columns
        '''
        # Perform encoding operations on self.df

    def perform_feature_engineering(self, response):
        '''
        input:
                  response: string of response name [optional argument that could be used for naming variables or index y column]

        output:
                  X_train: X training data
                  X_test: X testing data
                  y_train: y training data
                  y_test: y testing data
        '''
        # Perform feature engineering operations on self.df and return X_train, X_test, y_train, y_test

    def classification_report_image(self, y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
                y_train: training response values
                y_test:  test response values
                y_train_preds_lr: training predictions from logistic regression
                y_train_preds_rf: training predictions from random forest
                y_test_preds_lr: test predictions from logistic regression
                y_test_preds_rf: test predictions from random forest

        output:
                 None
        '''
        # Generate and save classification report

    def feature_importance_plot(self, model, X_data, output_pth):
        '''
        creates and stores the feature importances in pth
        input:
                model: model object containing feature_importances_
                X_data: pandas dataframe of X values
                output_pth: path to store the figure

        output:
                 None
        '''
        # Generate and save feature importance plot

    def train_models(self, X_train, X_test, y_train, y_test):
        '''
        train, store model results: images + scores, and store models
        input:
                  X_train: X training data
                  X_test: X testing data
                  y_train: y training data
                  y_test: y testing data
        output:
                  None
        '''
        # Train models and store results
