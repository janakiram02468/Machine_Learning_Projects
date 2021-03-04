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
data = pd.read_csv(r'C:\Users\JanakiRam\Documents\Machine Learning Projects\data_files\car_evaluation.csv')
data.head()

# Using labelEncoder converting catagorical data into Numerical data
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
data.buying = enc.fit_transform(data.buying)
data.maint = enc.fit_transform(data.maint)
data.lug_boot = enc.fit_transform(data.lug_boot)
data.safety = enc.fit_transform(data.safety)

# Splitting X and y
X = data.iloc[:,:-1]
y = data.outcome

X.head()

data.corr()

# Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)

# Building Random Forest Classifier model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=11,random_state=3,n_estimators=25)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(y_test, y_predict))
pd.crosstab(y_test, y_predict)


print(model.feature_importances_)


# from sklearn.externals import joblib
# joblib.dump(model,'car_trained.ml')



from sklearn.model_selection import GridSearchCV
parameters = {'max_depth':range(1,20),
              'random_state': range(0,20),
              'n_estimators':[10,15,20]
             }
grid = GridSearchCV(RandomForestClassifier(),parameters)
grid.fit(X_train,y_train)

