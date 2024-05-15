"""
The churn_library.py is a library of functions to find customers who are likely to churn.
"""
import argparse
import shap
import joblib
import logging
import pandas as pd
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

class ChurnLibrary:
    """
    A library of functions to analyze and predict customer churn in credit card services.

    Attributes:
        ALL_XDATA (int): Flag to indicate using all data.
        TRAIN_XDATA (int): Flag to indicate using training data.
        TEST_XDATA (int): Flag to indicate using test data.
    """


    ALL_XDATA = 0
    TRAIN_XDATA = 1
    TEST_XDATA = 2
    def __init__(self, file_path):
        """
        Initialize ChurnLibrary object.

        Parameters:
            file_path (str): Path to the CSV file containing data.
        """

        self.df = pd.read_csv(file_path, header=0, index_col=0)
        # first encode the Attrition_Flag since other categorical columns are depending on it.
        self.df['Churn'] = self.df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
        
        self.y = None # assigned when dataset is ready
        self.X = None # assigned when dataset is ready
        self.X_train = None # assigned when dataset is ready
        self.X_test = None # assigned when dataset is ready
        self.y_train = None # assigned when dataset is ready
        self.y_test = None # assigned when dataset is ready
        self.cv_rfc = None # models to be trained later
        self.rfc_best = None # the best estimator within cv_rfc
        self.lrc = None # models to be trained later
    
    def _save_plot(self, fig, save_name):
        """
        Save the plot to an image file.

        Parameters:
            fig (matplotlib.figure.Figure): Figure object to save.
            save_name (str): File path to save the image.
        """

        fig.savefig(save_name)
        plt.close(fig)

    
    def _plot_hist_or_bar(self, data, title, save_name, plot_type='hist'):
        """
        Plot and save histogram or bar plot.

        Parameters:
            data (pandas.Series): Data to plot.
            title (str): Title of the plot.
            save_name (str): File path to save the image.
            plot_type (str, optional): Type of plot ('hist' or 'bar'). Defaults to 'hist'.
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
        fig.tight_layout()
        self._save_plot(fig, save_name)
    
    def _plot_density(self, data, title, save_name):
        """
        Plot and save density plot.

        Parameters:
            data (pandas.Series): Data to plot.
            title (str): Title of the plot.
            save_name (str): File path to save the image.
        """

        fig, ax = plt.subplots(figsize=(20, 10))
        sns.histplot(data, stat='density', kde=True, ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        self._save_plot(fig, save_name)
    
    def _plot_heatmap(self, df, title, save_name):
        """
        Plot and save heatmap.

        Parameters:
            df (pandas.DataFrame): Dataframe to plot.
            title (str): Title of the plot.
            save_name (str): File path to save the image.
        """

        fig, ax = plt.subplots(figsize=(20, 10))
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2, ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        self._save_plot(fig, save_name)
    
    def perform_eda(self):
        """
        Perform exploratory data analysis (EDA) and save plots to images folder.
        """

        df = self.df
        self._plot_hist_or_bar(df['Attrition_Flag'], 
                               'Churned vs Active Customer', 
                               'images/churn_vs_active_histogram.png',
                              plot_type='bar')
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
        """
        Encode categorical columns by creating new columns with the proportion of churn for each category.

        Parameters:
            category_lst (list): List of column names containing categorical features.
            response (str): Name of the response column.

        Returns:
            df (pandas.DataFrame): DataFrame with new encoded columns.
        """

        
        for cat in category_lst:
            cat_groups = self.df.groupby(cat).mean()[response]
            
            self.df[f'{cat}_{response}'] = None
            for group in cat_groups.index:
                self.df.loc[self.df[cat]==group, f'{cat}_{response}'] = cat_groups.loc[group]

    def perform_feature_engineering(self, response, unwanted_cols=None):
        """
        Clean and prepare features for modeling.

        Parameters:
            response (str): Name of the response variable.
            unwanted_cols (list): List of columns to exclude from features.

        Returns:
            None
        """

        # Perform feature engineering operations on self.df and make X_train,
        # X_test, y_train, y_test
        # train test split 
        self.y = self.df[[response]]
        
        unwanted_cols = [response] if unwanted_cols is None else unwanted_cols + [response]
        self.X = self.df.drop(unwanted_cols, axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size= 0.3, random_state=42)
        self.y_train = self.y_train.values.ravel()
        self.y_test = self.y_test.values.ravel()

    def _save_classification_report(self, model_name, 
                                    y_test_pred, 
                                    y_train_pred, 
                                    save_path):
        """
        Generate and save a classification report for a given model.

        Parameters:
            model_name (str): Name of the model.
            y_test_pred (array-like): Predicted labels for test data.
            y_train_pred (array-like): Predicted labels for train data.
            save_path (str): File path to save the report.
        """

        assert self.y_test is not None, "y_test data is needed, use perform_feature_engineering to generate it"
        assert self.y_train is not None, "y_train data is needed, use perform_feature_engineering to generate it"
        fig, ax = plt.subplots(figsize=(5, 5))  # Create figure and axes

        # Format and add text to the axes
        ax.text(0.01, 1.25, f'{model_name} Train', {'fontsize': 10}, fontproperties='monospace')
        ax.text(0.01, 0.05, str(classification_report(self.y_test, y_test_pred)), {'fontsize': 10}, fontproperties='monospace')
        ax.text(0.01, 0.6, f'{model_name} Test', {'fontsize': 10}, fontproperties='monospace')
        ax.text(0.01, 0.7, str(classification_report(self.y_train, y_train_pred)), {'fontsize': 10}, fontproperties='monospace')
        ax.axis('off')  # Turn off axis

        self._save_plot(fig, save_path)

        
    def classification_report_image(
            self,
            y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf):
        """
        Generate classification reports for training and testing results and save as images.

        Parameters:
            y_train_preds_lr (array-like): Training predictions from logistic regression.
            y_train_preds_rf (array-like): Training predictions from random forest.
            y_test_preds_lr (array-like): Test predictions from logistic regression.
            y_test_preds_rf (array-like): Test predictions from random forest.

        Returns:
            None
        """

        # Generate and save classification report
        # scores
        # print(f'Best parameter combination: {self.cv_rfc.best_params_}')
        
        assert self.y_test is not None, "y_test data is needed, use perform_feature_engineering to generate it"
        assert self.y_train is not None, "y_train data is needed, use perform_feature_engineering to generate it"
        # store random forest report
        self._save_classification_report("Random Forest Classification",
                                         y_test_preds_rf,
                                         y_train_preds_rf,
                                         "./images/rfc_classification_report.png")

        self._save_classification_report("Logistic Regression",
                                         y_test_preds_lr,
                                         y_train_preds_lr,
                                         "./images/lrc_classification_report.png")
        
    def roc_curves_comparison(self, save_path):
        """
        Plot ROC curves for multiple models and save the plot.

        Parameters:
            save_path (str): Path to save the ROC plot image.

        Returns:
            None
        """

        assert self.X_test is not None, "X_test data is needed, use perform_feature_engineering to generate it"
        assert self.y_test is not None, "y_test data is needed, use perform_feature_engineering to generate it"

        fig, ax = plt.subplots(figsize=(15, 8))  # Create a figure and axes
        
        models = [self.rfc_best, self.lrc]
        # Plot ROC curve for each model
        for model in models:
            plot_roc_curve(model, self.X_test, self.y_test, ax=ax, alpha=0.8, name=f'{model.__class__.__name__} ROC Curve')

        # Add legend, reference line, and labels
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')
        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_ylabel('True Positive Rate (TPR)')
        ax.set_title('ROC Curves for Multiple Models')
        ax.legend()

        # Save the plot to the specified path
        plt.savefig(f"{save_path}/roc_curves.png")
        plt.close()  # Close the figure to release resources

    def scatter_plots_comparison(self, save_path):
        """
        Plot scatter plots of data points classified by multiple models and save the plot.

        Parameters:
            save_path (str): Path to save the scatter plot image.

        Returns:
            None
        """

        assert self.y_test is not None, "y_test data is needed, use perform_feature_engineering to generate it"
        assert self.X_test is not None, "X_test data is needed, use perform_feature_engineering to generate it"
        
        models = [self.rfc_best, self.lrc]
        num_models = len(models)
        num_classes = len(np.unique(self.y_test))

        fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5))

        for i, model in enumerate(models):
            ax = axes[i] if num_models > 1 else axes  # Select subplot

            # Make predictions on test data
            y_pred = model.predict(self.X_test)

            # Plot scatter plot for each class based on predicted labels
            for label in range(num_classes):
                indices = np.where(y_pred == label)[0]
                if indices.size > 0:  # Check if there are indices for this label
                    ax.scatter(indices, model.predict_proba(self.X_test.iloc[indices])[:, 1], label=f'Pred Class {label}', alpha=0.5, s=100)

            # Plot true classifications based on self.y_test
            for label in range(num_classes):
                indices = np.where(self.y_test == label)[0]
                if indices.size > 0:  # Check if there are indices for this label
                    ax.scatter(indices, model.predict_proba(self.X_test.iloc[indices])[:, 1], marker='x', label=f'True Class {label}', alpha=0.7)

                
            # Add decision line based on threshold
            ax.axhline(y=0.5, color='red', linestyle='--')
            ax.set_title(f'{model.__class__.__name__} Predictions vs. True Classes')
            ax.set_xlabel('Data Point Index')
            ax.set_ylabel('Probability of Positive Class')
            lgd = ax.legend(loc='upper left', bbox_to_anchor=(0.5,1.4))

        # Save the plot to the specified path
        plt.tight_layout()
        plt.savefig(f"{save_path}/scatterplot_comparison.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()  # Close the figure to release resources

    def feature_importance_plot(self, X_data, output_pth):
        """
        Create and save feature importances plot.

        Parameters:
            X_data (int): Flag to specify dataset (ALL_XDATA, TRAIN_XDATA, TEST_XDATA).
            output_pth (str): Path to save the plot.

        Returns:
            None
        """

        assert self.X is not None, "X data is needed, use perform_feature_engineering to generate it"
        assert self.X_test is not None, "X_test data is needed, use perform_feature_engineering to generate it"
        assert self.X_train is not None, "X_train data is needed, use perform_feature_engineering to generate it"

        if X_data==ChurnLibrary.ALL_XDATA:
            X_data = self.X
        elif X_data==ChurnLibrary.TRAIN_XDATA:
            X_data = self.X_train
        else:
            X_data = self.X_test
            
        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(self.rfc_best) # either pass the X_data as background data or it will use tree_path_dependent
        shap_values = explainer.shap_values(X_data)
        shap.summary_plot(shap_values, X_data, show=False) # or will create a bar plot because the output has more than one class
        plt.gcf() # get the current figure
        plt.tight_layout()
        plt.savefig(f"{output_pth}/feature_summaryplot_bar_multiclass.png")
        plt.close()
        # refer to https://github.com/shap/shap/issues/837#issuecomment-539491822
        shap.summary_plot(shap_values[0], X_data, show=False) # one question is that which of [0] or [1] represent churn=0?
        plt.gcf() # get the current figure
        plt.tight_layout()
        plt.savefig(f"{output_pth}/feature_summaryplot_dot.png")
        plt.close()
        
        # now lets get the feature importance according to the gridsearch
        # which is based on the decrease in self.cv_rfc performance.
        # while SHAP is based on the magnitude of feature attributions.
        
        # Calculate feature importances
        importances = self.rfc_best.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [self.X.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20,5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(self.X.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(self.X.shape[1]), names, rotation=90)
        plt.tight_layout()
        plt.savefig(f"{output_pth}/feature_gridsearch.png")
        plt.close()
    
    def _train_random_forest_classifier_with_grid_search(self, X_train, y_train, param_grid):
        """
        Train a Random Forest classifier with Grid Search.

        Parameters:
            X_train (array-like): Training features.
            y_train (array-like): Training labels.
            param_grid (dict): Parameter grid for GridSearchCV.

        Returns:
            None
        """

        rfc = RandomForestClassifier(random_state=42)
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)
        return cv_rfc
    
    def _train_logistic_regression(self, X_train, y_train, solver='lbfgs', max_iter=3000):
        """
        Train a Logistic Regression classifier.

        Parameters:
            X_train (array-like): Training features.
            y_train (array-like): Training labels.
            solver (str): Solver to use (default='lbfgs').
            max_iter (int): Maximum number of iterations (default=3000).

        Returns:
            None
        """

        lrc = LogisticRegression(solver=solver, max_iter=max_iter)
        lrc.fit(X_train, y_train)
        return lrc
    
    def train_models(self, pretrained=False):
        """
        Train models, store results and models.

        Raises:
            ValueError: If necessary data attributes are not initialized.
        """

        # Perform feature engineering if needed
        # X_train, X_test, y_train, y_test = self.perform_feature_engineering('Churn')
        
        # Grid search parameters
        param_grid = { 
            'n_estimators': [200, 500],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [4, 50, 100],
            'criterion': ['gini', 'entropy']
        }
        if pretrained:
            rfc_model_path = os.path.join('./models/', 'rfc_model.pkl')
            lrc_model_path = os.path.join('./models/', 'logistic_model.pkl')
            assert os.path.exists(rfc_model_path), "Random Forest model not saved"
            assert os.path.exists(lrc_model_path), "Logistic Regression model not saved"
            # load the pre-trained models
            self.rfc_best = joblib.load(rfc_model_path)
            self.lrc = joblib.load(lrc_model_path)
        else:
            # train Random Forest
            self.cv_rfc = self._train_random_forest_classifier_with_grid_search(self.X_train, self.y_train, param_grid)
            self.rfc_best = self.cv_rfc.best_estimator_

            # train Logistic Regression
            self.lrc = self._train_logistic_regression(self.X_train, self.y_train)
            # save best model
            joblib.dump(self.rfc_best, './models/rfc_model.pkl')
            joblib.dump(self.lrc, './models/logistic_model.pkl')
        # generate predictions
        y_train_preds_rf = self.rfc_best.predict(self.X_train)
        y_test_preds_rf = self.rfc_best.predict(self.X_test)
        y_train_preds_lr = self.lrc.predict(self.X_train)
        y_test_preds_lr = self.lrc.predict(self.X_test)

if __name__ == '__main__':

    # Set up logging
    logging.basicConfig(
        filename='./logs/churn_library.log',
        level=logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s'
    )

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Import data for churn prediction.')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the CSV data file.')

    # Parse command-line arguments
    args = parser.parse_args()

    # 1. import the data
    cl_object = ChurnLibrary(args.data_path)
    img_pth = './images'
    
    # 2. perform exploratory data analysis
    try:
        cl_object.perform_eda()
        logging.info("EDA files stored successfully")
    except Exception as err:
        logging.error(f"problem occured during EDA analysis: {err}" , exc_info = err)
        raise err
    # 3. encode category columns to numbers
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    try:
        cl_object.encoder_helper(cat_columns, 'Churn')
        logging.info("selected columns are encoded successfully!")
    except Exception as err:
        logging.error(f"Failed to encode selected columns: {err}" , exc_info = err)
        raise err
        
    # 4. create training and test data sets
    removing_cols = cat_columns + ['Attrition_Flag']
    try:
        cl_object.perform_feature_engineering('Churn', removing_cols)
        logging.info("training and test datasets are created successfully!")
    except Exception as err:
        logging.error(f"Failed to split the dataset into training and test: {err}" , exc_info = err)
        raise err
    
    # 5. train two models, regression and randomforest
    try:
        cl_object.train_models(False)
        logging.info("the models trained successfully!")
    except Exception as err:
        logging.error(f"Failed to train the model on the prepared datasets: {err}" , exc_info = err)
        raise err
    
    # 6. generate classification reports for both regression and randomforest models
    # Generate predictions for logistic regression (0 or 1) with a threshold of 0.5
    y_train_preds_lr = cl_object.lrc.predict(cl_object.X_train)
    y_test_preds_lr = cl_object.lrc.predict(cl_object.X_test)

    # Generate predictions for random forest (0 or 1) with a threshold of 0.5
    y_train_preds_rf = cl_object.rfc_best.predict(cl_object.X_train)
    y_test_preds_rf = cl_object.rfc_best.predict(cl_object.X_test)
    
    try:
        cl_object.classification_report_image(y_train_preds_lr, y_train_preds_rf, 
                                          y_test_preds_lr, y_test_preds_rf)
        logging.info("classification reports created successfully!")
    except Exception as err:
        logging.error(f"Failed creating classification reports: {err}" , exc_info = err)
        raise err
        
    # 7. generate roc curves for both models
    try:
        cl_object.roc_curves_comparison(img_pth)
        logging.info("roc curve analysis conducted successfully!")
    except Exception as err:
        logging.error(f"Failed conducting roc curve analysis: {err}" , exc_info = err)
        raise err
        
    # 8. generate scatter plot comparison for both models
    try:
        cl_object.scatter_plots_comparison(img_pth)
        logging.info("data points scatter plot created successfully!")
    except Exception as err:
        logging.error(f"Failed creating data points scatter plot: {err}" , exc_info = err)
        raise err
    
    # 9. generate feature importance plot for random forest model
    try:
        cl_object.feature_importance_plot(ChurnLibrary.TEST_XDATA, img_pth)
        logging.info("data points scatter plot created successfully!")
    except Exception as err:
        logging.error(f"Failed creating data points scatter plot: {err}" , exc_info = err)
        raise err