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
import pandas as pd
data = pd.read_csv(r'C:\Users\JanakiRam\Documents\Machine Learning Projects\data_files\car_evaluation.csv')
data.head()


from collections import Counter
Counter(data.outcome)


# Using labelEncoder converting catagorical data into Numerical data
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
data.buying = enc.fit_transform(data.buying)
data.maint = enc.fit_transform(data.maint)
data.lug_boot = enc.fit_transform(data.lug_boot)
data.safety = enc.fit_transform(data.safety)
data.head()

# Splitting X and y
X = data.iloc[:,:-1]
y = data.outcome

# Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=10)




# Building KNN model
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)


# Finding the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_predict))
pd.crosstab(y_test,y_predict)


