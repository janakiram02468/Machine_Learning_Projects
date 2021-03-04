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

from sklearn.feature_extraction.text import CountVectorizer



messages = ['call you tonight', 'Call me a cab', 'please call me.. please']



# instantiate CountVectorizer (vectorizer)
vect = CountVectorizer()
vect.fit(messages)
vect.get_feature_names()



messages_transformed = vect.transform(messages)
print(messages)
print(vect.get_feature_names())
messages_transformed.toarray()





data = pd.DataFrame(messages_transformed.toarray())
data.columns = vect.get_feature_names()
print(messages)
data.head()





data.loc[0,'outcome'] ='info'
data.loc[1,'outcome'] ='order'
data.loc[2,'outcome'] = "request"




data.head()


