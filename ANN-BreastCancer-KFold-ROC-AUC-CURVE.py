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
data.head()



from sklearn.preprocessing import scale, StandardScaler
X = data.iloc[:,:-1]
X = pd.DataFrame(scale(X))
## X_new= scale_model.transform(X_new)  transform new data
y = data.outcome

# Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)


X.head(2)

data.head(2)




from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=6)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(accuracy_score(y_test, y_predict))
pd.crosstab(y_test, y_predict)





from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(scores)
print(scores.mean())


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for breast cancer classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()
print(metrics.roc_auc_score(y_test,y_predict))
thresholds

