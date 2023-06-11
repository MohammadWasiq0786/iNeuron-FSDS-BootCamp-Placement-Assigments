"""
# Submitted by: Mohammad Wasiq

## Email: gl0427@myamu.ac.in

# Placement Machine Learning Assignment 10
"""

'''
Q10. An Ad- Agency analyzed a dataset of online ads and used a machine learning
model to predict whether a user would click on an ad or not.

Dataset:- https://www.kaggle.com/c/avazu-ctr-prediction
'''

# Ans:-

## Import the required libraries:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

## Load the dataset into a Pandas DataFrame:-
data = pd.read_csv('train.csv')


## Data preprocessing:
### Remove unnecessary columns:

data.drop(['id', 'hour', 'device_id', 'device_ip'], axis=1, inplace=True)

### Encode categorical features:

label_encoders = {}
categorical_features = ['site_id', 'site_domain', 'app_id', 'app_domain', 'device_model']

for feature in categorical_features:
    label_encoders[feature] = LabelEncoder()
    data[feature] = label_encoders[feature].fit_transform(data[feature].astype(str))


## Split the data into training and testing sets:
X = data.drop('click', axis=1)
y = data['click']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## Train a Random Forest classifier:
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)


## Make predictions on the test set and evaluate the model:
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
