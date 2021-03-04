# To ignore the warnings

import warnings
warnings.filterwarnings('ignore')

# importing necessory libraries
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# matplotlib is for visualization purpose
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 10,6

#Using Naive Bayes to predict spam
#Use Latin encoding as the Data has non UTF-8 Chars
data = pd.read_csv(r"C:\Users\JanakiRam\Documents\Machine Learning Projects\data_files\spam.csv",encoding='latin-1')

data.shape
data.head()


from collections import Counter
Counter(data.type)

# Splitting X and y
X =  data.email
y = data.type
X.head()


#Vectorization : Transforming TEXT to Vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
feature_names = vectorizer.get_feature_names()

# Lenght of feature names
print(len(feature_names))
feature_names[2000:2010]

# converting the data into Array
X = X.toarray()
print(X.shape)
print(y.shape)

# Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)


#Fitting Naive Bayes algo
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
model = BernoulliNB()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)



print(accuracy_score(y_test,y_predict))
print(classification_report(y_test,y_predict))
pd.crosstab(y_test,y_predict)


#Checking new email for spam

#NewEmail = pd.Series(["Hi team, We have meeting tomorrow"])
NewEmail = pd.Series(['**FREE MESSAGE**Thanks for using the Auction Subscription Service. 18 . 150p/MSGRCVD 2 Skip an Auction txt OUT. 2 Unsubscribe txt STOP CustomerCare 08718726270'])

NewEmail_transformed = vectorizer.transform(NewEmail)


# Predinting the new email either spam or ham
print(model.predict(NewEmail_transformed))



