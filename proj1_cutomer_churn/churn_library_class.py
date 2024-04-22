'''
The churn_library.py is a library of functions to find customers who are likely to churn.
'''

import pandas as pd
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

class ChurnLibrary:
    def __init__(self, file_path):
        """
        Initialize ChurnLibrary object.

        Parameters:
        - file_path (str): Path to the CSV file containing data.
        """
        self.df = pd.read_csv(file_path, header=0)
        self.df['Churn'] = self.df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    
    def _save_plot(self, fig, save_name):
        """
        Save the plot to an image file.

        Parameters:
        - fig (matplotlib.figure.Figure): Figure object to save.
        - save_name (str): File path to save the image.
        """
        plt.savefig(save_name)
        plt.close()
    
    def _plot_hist_or_bar(self, data, title, save_name, plot_type='hist'):
        """
        Plot and save histogram or bar plot.

        Parameters:
        - data (pandas.Series): Data to plot.
        - title (str): Title of the plot.
        - save_name (str): File path to save the image.
        - plot_type (str, optional): Type of plot ('hist' or 'bar'). Defaults to 'hist'.
        """
        fig, ax = plt.subplots(figsize=(20, 10))
        if plot_type == 'hist':
            ax.hist(data)
        elif plot_type == 'bar':
            data.value_counts().plot(kind='bar', ax=ax)
        else:
            raise ValueError("Invalid plot_type. Use 'hist' or 'bar'.")
        ax.set_title(title)
        ax.set_ylabel('Number of Customers')
        self._save_plot(fig, save_name)
    
    def _plot_density(self, data, title, save_name):
        """
        Plot and save density plot.

        Parameters:
        - data (pandas.Series): Data to plot.
        - title (str): Title of the plot.
        - save_name (str): File path to save the image.
        """
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.histplot(data, stat='density', kde=True, ax=ax)
        ax.set_title(title)
        self._save_plot(fig, save_name)
    
    def _plot_heatmap(self, df, title, save_name):
        """
        Plot and save heatmap.

        Parameters:
        - df (pandas.DataFrame): Dataframe to plot.
        - title (str): Title of the plot.
        - save_name (str): File path to save the image.
        """
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2, ax=ax)
        ax.set_title(title)
        self._save_plot(fig, save_name)
    
    def perform_eda(self):
        """
        Perform exploratory data analysis (EDA) and save plots to images folder.
        """
        df = self.df
        self._plot_hist_or_bar(df['Attrition_Flag'], 
                               'Churned vs Active Customer', 
                               'images/churn_vs_active_histogram.png')
        self._plot_hist_or_bar(df['Customer_Age'], 
                               'Customer Age', 
                               'images/customer_age_histogram.png')
        self._plot_hist_or_bar(df['Marital_Status'], 
                               'Marital Status', 
                               'images/marital_status_barplot.png', 
                               plot_type='bar')
        self._plot_density(df['Total_Trans_Ct'], 
                           'Total Transaction Count in The Last 12 Months', 
                           'images/density_total_trans_count.png')
        self._plot_heatmap(df, 
                           'Heatmap Correlation', 
                           'images/heatmap_corr.png')

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
        # Perform feature engineering operations on self.df and return X_train,
        # X_test, y_train, y_test

    def classification_report_image(
            self,
            y_train,
            y_test,
            y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf):
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
if __name__ == 'main':
    cl_object = ChurnLibrary("./data/bank_data.csv")
    