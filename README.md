# Churn Analysis

A simple churn analysis project that explores a customer dataset, performs exploratory data analysis (EDA), and provides a starting point for building churn prediction models. The primary notebook is `Churn_Analysis.ipynb` developed for Google Colab.

## Table of Contents
- Overview
- Dataset
- Notebook structure
- Requirements
- How to run (Colab / locally)
- Recommended next steps
- Contact

## Overview
This project analyzes customer behavior to identify patterns associated with churn. The included notebook loads customer data, inspects and visualizes key features, and contains code scaffolding for model building and evaluation.

## Dataset
The notebook expects a CSV dataset named `customer_data.csv`. Example columns found in the dataset:
- year
- customer_id
- phone_no
- gender
- age
- no_of_days_subscribed
- multi_screen
- mail_subscribed
- weekly_mins_watched
- minimum_daily_mins
- maximum_daily_mins
- weekly_max_night_mins
- videos_watched
- maximum_days_inactive
- customer_support_calls
- churn

Note: In the notebook the dataset is loaded from Google Drive at:
`/content/drive/My Drive/Colab Notebooks/customer_data.csv`
Make sure to place your CSV there when running in Colab, or update the path in the notebook.

## Notebook structure (Churn_Analysis.ipynb)
1. Import Google Drive (for Colab)
2. Import libraries (pandas, numpy, matplotlib, seaborn, sklearn.metrics, etc.)
3. Read dataset into a pandas DataFrame
4. Exploratory Data Analysis (head, tail, basic inspection, visualizations)
5. (Placeholder) Preprocessing and feature engineering
6. (Placeholder) Model training and evaluation

The notebook currently includes EDA steps and metric imports. You can extend it to add preprocessing, model training (e.g., logistic regression, random forest), cross-validation, hyperparameter tuning, and in-depth evaluation.

## Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- (Optional, if using Colab) Google Colab environment with Google Drive access

You can install required packages with pip:
pip install pandas numpy matplotlib seaborn scikit-learn

## How to run

Running in Google Colab (recommended):
1. Open `Churn_Analysis.ipynb` in Colab.
2. Mount Google Drive (the notebook includes code to mount).
3. Upload `customer_data.csv` to:
   `/content/drive/My Drive/Colab Notebooks/customer_data.csv`
   or update the path in the notebook to point to your CSV.
4. Run notebook cells sequentially.

Running locally:
1. Clone the repository.
2. Place `customer_data.csv` in the project directory (or update the path in the notebook).
3. Install required Python packages.
4. Open the notebook in Jupyter and run cells.

## Recommended next steps
- Clean missing values and handle inconsistent entries (e.g., missing gender, NaNs).
- Convert categorical variables to numeric (one-hot encoding or label encoding).
- Feature engineering: aggregate usage, recency, frequency metrics, tenure buckets.
- Train and compare classification models: logistic regression, decision trees, random forest, XGBoost.
- Evaluate models with appropriate metrics for imbalanced data (ROC AUC, precision-recall, F1, confusion matrix).
- Add cross-validation and hyperparameter tuning (GridSearchCV / RandomizedSearchCV).
- Save the final model and create scripts for batch or real-time inference.

## License
This repository does not include a license file. If you plan to share the project, consider adding an open-source license (e.g., MIT, Apache-2.0).

## Contact
Repo owner: ABHINAV-2408

If you want, I can:
- Draft a requirements.txt
- Add a license file
- Extend the notebook with preprocessing and model training code
- Create a small script (train.py) to run training outside the notebook
