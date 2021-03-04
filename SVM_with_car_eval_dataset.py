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

# importing the data
data = pd.read_csv(r'C:\Users\JanakiRam\Documents\Machine Learning Projects\data_files\car_evaluation.csv')
data.head(2)

# Splitting X and y
X = data.iloc[:,:-1]
y = data.outcome

# Using labelEncoder converting catagorical data into Numerical data
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
X.buying = enc.fit_transform(X.buying)
X.maint = enc.fit_transform(X.maint)
X.lug_boot = enc.fit_transform(X.lug_boot)
X.safety = enc.fit_transform(X.safety)

# splitting train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,
                                                   random_state=10)

# Build SVC a model
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
model = SVC(kernel='rbf', C = 10, gamma = 0.5)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(accuracy_score(y_test,y_predict))
pd.crosstab(y_test, y_predict)

