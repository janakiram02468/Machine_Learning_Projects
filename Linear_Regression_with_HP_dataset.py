# To ignore the warnings

import warnings
warnings.filterwarnings('ignore')

# importing necessory libraries
import pandas as pd
import numpy as np

# matplotlib is for visualization purpose
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 10,6


# importing the dataset
data = pd.read_csv(r'C:\Users\JanakiRam\Documents\Machine Learning Projects\data_files\hp_data.csv')
data.head(3)

# Taking X variables which is independent variables and y is output variable

X = data.loc[:,['sqft','totalFloor','yearsOld','bhk']]
y = data.price
X.head()

# Splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                   test_size=0.3,
                                                   random_state=10)
print("X_train Shape :", X_train.shape)
print("X_test Shape :", X_test.shape)
print("y_train Shape :", y_train.shape)
print("y_test Shape :", y_test.shape)

# Applying the Linear Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)  # Training
y_predict = model.predict(X_test) # Predict for Test

# predicting the output
from sklearn.metrics import r2_score
print(r2_score(y_test, y_predict))



