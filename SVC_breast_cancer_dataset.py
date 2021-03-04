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
data.head(2)


# Scaling the Data i.e normalizing
from sklearn.preprocessing import scale
X = data.iloc[:,:-1]
X = scale(X)
y = data.outcome

# Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,
                                                   random_state=10)
# Building SVC model
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
model = SVC(kernel='rbf', C = 7, gamma=0.001)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(accuracy_score(y_test,y_predict))
print(pd.crosstab(y_test, y_predict))

