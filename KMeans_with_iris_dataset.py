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

# Splitting X and y
X = data.iloc[:,:-1]
X.head(1)



# Build KMeans model
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=10)
model.fit(X)

print(model.labels_)
print(np.array(data.target))
