1) Customer Churn Prediction

This project aims to predict whether a customer will churn (leave the company) or stay, based on their demographic, account, and service-related information. By analyzing customer behavior patterns, the model helps businesses take proactive steps to retain valuable customers.

2) Project Overview

Customer churn is one of the major problems faced by subscription-based businesses. This project builds a machine learning classification model that identifies customers likely to churn, enabling companies to target them with retention strategies.

3) Objective

Analyze customer data and identify key features affecting churn.

Preprocess and clean data for model training.

Train multiple classification models and compare their performance.

Predict churn with the most accurate model.

4) Dataset

The dataset contains various customer-related attributes such as:

CustomerID – Unique identifier for each customer

Gender

Age

Tenure – Number of months the customer has stayed with the company

Balance – Account balance

Products/Services – Number of products the customer uses

IsActiveMember – Whether the customer is active

Exited – Target variable (1 = churned, 0 = retained)

5) Tech Stack

Languages: Python 

Libraries Used:

numpy  
pandas  
matplotlib  
seaborn  
scikit-learn  
xgboost

6) Model Building

Data Preprocessing

Handled missing values

Performed label encoding for categorical variables

Applied feature scaling using StandardScaler

Model Training

Trained multiple models including:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

XGBoost Classifier

Model Evaluation

Compared accuracy, precision, recall, and F1-score

Used confusion matrix and ROC-AUC curve for performance evaluation

7) Results
Model	Accuracy	Precision	Recall	F1-Score

Logistic Regression	83%	0.81	0.79	0.80

Decision Tree	86%	0.84	0.83	0.83

Random Forest	89%	0.87	0.88	0.88

XGBoost	91%	0.90	0.91	0.91

✅ XGBoost performed the best with the highest accuracy and generalization performance.

8) Visualizations

Churn distribution

Correlation heatmap

Feature importance plot

Confusion matrix
