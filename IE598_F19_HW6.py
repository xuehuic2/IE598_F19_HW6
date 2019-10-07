#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('ccdefault.csv')


# In[3]:


df.shape


# In[4]:


df = df[df.columns[1:]]


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


#no need to drop na
#decision tree, 
import sklearn
from sklearn.model_selection import train_test_split

X = df.iloc[:, 0:23].values
y=df[df.columns[23]]

#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.10, random_state=1)


# In[8]:


dX = df.iloc[:, 0:23]
df.feature_names = list(dX.columns.values) 
df.class_names = df.columns[23]


# In[9]:


df.class_names


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=1)
# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth of
dt = DecisionTreeClassifier(max_depth=3, random_state=0)

# Fit dt to the training set
dt.fit(X_train, y_train)

# Import accuracy_score
from sklearn.metrics import accuracy_score

# Predict test set labels
y_pred = dt.predict(X_test)

# Compute test set accuracy  
acc = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))


# In[11]:


from sklearn.externals.six import StringIO  
dot_data = StringIO()


# In[12]:


from IPython.display import Image  


# In[13]:


from sklearn.tree import export_graphviz


# In[14]:


import pydotplus


# In[15]:


export_graphviz(dt, out_file=dot_data,feature_names = df.feature_names, class_names = df.class_names,   
                filled=True, rounded=True,
                special_characters=True)


# In[16]:


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[18]:


resulttrain = []
resulttest = []

#for x in range(1,13):
#    result.append((x, Boston_monthly_temp(x)))
for k in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=k)
    dt = DecisionTreeClassifier(max_depth=3, random_state=k)
    # Fit dt to the training set
    dt.fit(X_train, y_train)

    y_train_pred = dt.predict(X_train)
    y_test_pred = dt.predict(X_test)
    
    acctrain = accuracy_score(y_train, y_train_pred)
    #print("random state =",k ,"Train set accuracy: {:.4f}".format(acctrain))
    
    resulttrain.append(( k,acctrain))
    acctest = accuracy_score(y_test, y_test_pred)
   # print("random state =",k ,"Test set accuracy: {:.4f}".format(acctest))
    resulttest.append(( k,acctest))


# In[19]:


resulttrain


# In[20]:


resulttest


# In[26]:


X = df.iloc[:, 0:23].values
y=df[df.columns[23]]

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
skf = StratifiedKFold(n_splits=10,random_state=1)
skf.get_n_splits(X, y)
for train_index, test_index in skf.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

clf = tree.DecisionTreeClassifier(max_depth=3)
insample_scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, n_jobs=-1)
outsample_scores = cross_val_score(estimator=clf, X=X_test, y=y_test, cv=10, n_jobs=-1)
insample_scores


# In[27]:


insample_scores.mean()


# In[28]:


insample_scores.std()


# In[29]:


outsample_scores


# In[30]:


outsample_scores.mean()


# In[31]:


outsample_scores.std()


# In[32]:


print("My name is Xuehui Chao")
print("My NetID is: xuehuic2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:




