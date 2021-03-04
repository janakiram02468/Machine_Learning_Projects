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
data  = pd.read_csv(r'C:\Users\JanakiRam\Documents\Machine Learning Projects\data_files\wine_quality_class.csv')
data.head()

# Splitting X and y
X = data.iloc[:,:-2]
y = data.Quality

# Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)


# Building XGB Classifier model
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
model = XGBClassifier(max_depth=6, n_estimators=100,learning_rate=0.75,
                     random_state=20)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(accuracy_score(y_test, y_predict))
pd.crosstab(y_test, y_predict)
