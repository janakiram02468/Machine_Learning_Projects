# To ignore the warnings

import warnings
warnings.filterwarnings('ignore')

# importing necessory libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# matplotlib is for visualization purpose
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 10,6

# importing the dataset
data = pd.read_csv(r'C:\Users\JanakiRam\Documents\Machine Learning Projects\data_files\breast_cancer.csv')
data.shape

# Using info(), it will show the how many attributes present in the dataset
data.info()

# Using describe(), it will tell us the mean and some other information
data.describe()

# Splitting X and y
X = data.iloc[:,:-1]
y = data.outcome
print("X shape",X.shape)
print("y shape",y.shape)


# Splitting data into Train and Test sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                   test_size=0.3,
                                                   random_state=10)
print("X_train Shape :", X_train.shape)
print("X_test Shape :", X_test.shape)
print("y_train Shape :", y_train.shape)
print("y_test Shape :", y_test.shape)


# Building Logistic Regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)


# Finding the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_predict))

# Confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_predict))
print(pd.crosstab(y_test, y_predict))

# Logisticregression has term regression but its actually a binary classifier.
# We have regression term as actual output of Logistic equation (sigmoid function ) 1/1+e-z, is continuous.
