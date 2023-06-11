"""
# Submitted by: Mohammad Wasiq

## Email: gl0427@myamu.ac.in

# Placement Advanced Machine Learning Assignment 1
"""

'''
Q1. An Ad- Agency analyzed a dataset of online ads and used a machine learning
model to predict whether a user would click on an ad or not.

Dataset:- https://www.kaggle.com/c/avazu-ctr-prediction
'''

# Ans:

import dask.dataframe as dd
import mlflow
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LogisticRegression
from dask_ml.metrics import accuracy_score

# Read the dataset using Dask
df = dd.read_csv('train.csv')

# Split the data into features (X) and target (y)
X = df.drop('friend_request_accepted', axis=1)
y = df['friend_request_accepted']

# Split the data into training and testing sets using Dask's train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Log the metrics using MLflow
with mlflow.start_run():
    mlflow.log_param('model', 'Logistic Regression')
    mlflow.log_metric('accuracy', accuracy)
