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
data = pd.read_csv(r'C:\Users\JanakiRam\Desktop\ML_datamites\breast_cancer.csv')
data.head()

# Splitting X and y
X = data.iloc[:,:-1]

# Scaling the dataset
from sklearn.preprocessing import scale
X_scaled = pd.DataFrame(scale(X))

X_scaled.columns = X.columns
y = data.outcome
X.head(2)

# Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.2,
                                                    random_state=10)



# Building Decision Tree model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=4,random_state=0)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_predict))
print(pd.crosstab(y_test, y_predict))


