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


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

# importing the dataset
data = pd.read_csv(r'C:\Users\JanakiRam\Documents\Machine Learning Projects\data_files\iris.csv')
data.head()


# In[5]:


X = data.iloc[:,:-1]
y = data.target
X.head()


# In[23]:


#Defining and fitting
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=2)
model.fit(X,y)


# In[24]:


#Visualizing
dot_data = StringIO()
export_graphviz(model, out_file=dot_data,  
                filled=True, 
                feature_names = X.columns,
                class_names=['setasa','versi color','virginca'],
                rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[12]:


X.head()


# In[13]:


data.head()


# In[14]:


data[data.petal_length <=4.0].target.value_counts()


# In[ ]:





# In[ ]:




