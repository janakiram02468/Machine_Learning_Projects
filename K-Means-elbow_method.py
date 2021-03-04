# importing necessory libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# matplotlib is for visualization purpose
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 10,6
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn.metrics import confusion_matrix, classification_report
plt.figure(figsize=(10,6))

# importing the dataset
iris = pd.read_csv(r'C:\Users\JanakiRam\Documents\Machine Learning Projects\data_files\iris.csv')
iris.head()

# Splitting X and y

x = iris.iloc[:,:-1]
y = iris.target
x.head()

# Applying the model
clustering = KMeans(n_clusters=3, random_state=1)
clustering.fit(x)

print(clustering.labels_)


#plotting your model outputs

color_theme = np.array(['red','blue','green'])

plt.subplot(1,2,1)
plt.scatter(x=iris.petal_length, y=iris.petal_width, c= color_theme[iris.target],s=50)
plt.title ("This is Actual Flower Cluster")

color_theme2 = np.array(['green','red','blue','orange'])

plt.subplot(1,2,2)
plt.scatter(x=iris.petal_length, y=iris.petal_width, c= color_theme2[clustering.labels_],s=50)
plt.title ("This is KMeans Clustering ")

print(clustering.cluster_centers_)

target_predicted = np.choose(clustering.labels_,[2,0,1]).astype(np.int64)
target_predicted

confusion_matrix(iris.target,target_predicted)


x.shape
X = iris.iloc[:,[2,3]]

X.shape[0]

# clustering dataset
# determine k using elbow method
 
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

# create new plot and data
X = iris.iloc[:,[2,3]]
 
# k means determine k
distortions = []
K = range(1,17)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
 
# Plot the elbow
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.set_xticks(range(1,25))
ax.set_xlabel('k')
ax.set_ylabel('Distortion')
ax.set_title('The Elbow Method showing the optimal k')
ax.plot(K, distortions, marker = 'x')

plt.show()

