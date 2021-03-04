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

# Scaling
from sklearn.preprocessing import scale
X = data.iloc[:,:-1]

# Splitting X and y
X = scale(X)
y = data.outcome

# Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,
                                                   random_state=10)


print(X_train.shape)
print(X_test.shape)

# Building a MLP classifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(learning_rate_init=0.01,
                      hidden_layer_sizes=(5,56,89),
                      random_state=9)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(accuracy_score(y_test,y_predict))
pd.crosstab(y_test, y_predict)




from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(learning_rate_init=0.1,
                      learning_rate= 'constant',
                      hidden_layer_sizes=(59),
                      random_state=10)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(accuracy_score(y_test,y_predict))
pd.crosstab(y_test, y_predict)








# In[ ]:




