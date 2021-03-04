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
data = pd.read_csv(r'C:\Users\JanakiRam\Documents\Machine Learning Projects\data_files\iris.csv')
X = data.iloc[:,:-1]
y = data.target
X.head(2)

# Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
                                                   random_state=10)
# Building Random Forest model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(accuracy_score(y_test, y_predict))
pd.crosstab(y_test, y_predict)



# Building PCA model

from sklearn.decomposition import PCA
pca = PCA()

X.head(1)

X_pca = pd.DataFrame(pca.fit_transform(X),columns=['pc1','pc2','pc3','pc4'])
X_pca.head(1)


print(pca.explained_variance_ratio_)

X_pca_selected = X_pca.iloc[:,:2]
X_pca_selected.head(2)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pca_selected,y,test_size=0.3,
                                                   random_state=10)


X_train.head(1)

# After applying PCA introducing the RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(accuracy_score(y_test, y_predict))
pd.crosstab(y_test, y_predict)

