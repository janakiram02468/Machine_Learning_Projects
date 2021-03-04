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


# Importing data files
data = pd.read_csv(r'C:\Users\JanakiRam\Documents\Machine Learning Projects\data_files\wine_quality_class.csv')
data.head()


from collections import Counter
Counter(data.Quality)


# Checking the correlation between the variables
data_corr  = data.corr()
data_corr.loc[:,'quality rate'].sort_values()

# Splitting the X and y
X = data.iloc[:,0:-2]
y = data.Quality

# Scaling the data
from sklearn.preprocessing import scale
X = scale(X)

# Splitting the train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=7)

# Building the KNN model
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=6)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
print(accuracy_score(y_test, y_predict))
pd.crosstab(y_test, y_predict)


# Using grid search to find the n_neighbors value
import numpy as np
from sklearn.model_selection import GridSearchCV
parameters = { 'n_neighbors': range(1,100) }
grid = GridSearchCV(KNeighborsClassifier(),parameters)
grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_params_)

