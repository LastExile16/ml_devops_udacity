# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The project addresses the issue of increasing customer churn in a bank's credit card services by providing predictive insights to identify potential churners. The goal is to enable proactive intervention to improve service quality and retain customers.

In this project, we implement two predictive models, logistic regression, and random forest, to forecast customer churn. The models are evaluated and compared based on their performance, with the random forest model demonstrating superior predictive capabilities.

Through detailed analysis, the project identifies key characteristics associated with churning customers, notably emphasizing low transaction count and low revolving balance as a significant indicators.
## Dependencies
This project depends on multiple packages. 

To be able to run the project you can install the required packages with the following command.
```bash
python -m pip install -r requirements_py3.6.txt
```

## Files and Data Description
- **`churn_library_class.py`**: This file contains a Python class that implements logistic regression and random forest models for churn prediction. The class provides various functions for data analysis and feature extraction using SHAP (SHapley Additive exPlanations) analysis.

- **`churn_script_logging_and_tests.py`**: This script includes test functions for the `churn_library_class.py` file. It utilizes dummy data to validate the functionality of each method within the class.

## Running Files
To execute the predictive models, follow these steps:
1. Prepare the dataset path.
2. Pass the dataset path using the `--data-path` flag to the `churn_library_class.py` file:

```bash
    python churn_library_class.py --data-path ./data/bank_data.csv
```
   

To run the test script using pytest from the terminal:
```bash
pytest churn_script_logging_and_tests.py
```

These commands allow for seamless execution of the models and test functions, providing insights into customer churn prediction and model evaluation.
