"""
# Submitted by: Mohammad Wasiq

## Email: gl0427@myamu.ac.in

# Placement Machine Learning Assignment 9
"""

"""
Q9. A cyber security agent wants to check the Microsoft Malware so need he came
to you as a Machine learning Engineering with Data, You need to find the Malware
using a supervised algorithm and you need to find the accuracy of the model.

Dataset:- https://www.kaggle.com/competitions/microsoft-malware-prediction/data
"""

# Ans:-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("train.csv")

# Preprocessing
# Select the features and target variable
features = data.drop(["MachineIdentifier", "HasDetections"], axis=1)
target = data["HasDetections"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)