
# coding: utf-8

# In[1]:


# importing libraries
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython.display import Image 


# In[2]:


# data reading
playstoredata = pd.read_csv('googleplaystore.csv')


# In[3]:


#dropping null values
playstoredata.dropna(axis = 0,inplace = True)


# In[4]:


# funciton to modify column Installs
def Install(installs):
    installs = installs.strip('+')
    installs = installs.replace(',','')
    return int(installs)


# In[5]:


# calling funciton
playstoredata['Installs'] = playstoredata['Installs'].apply(lambda ins:Install(ins))
playstoredata.dropna(axis =0,inplace = True)


# In[6]:


# function to modify each value in Size column
def Size(size):
    if  'M' in size:
        size =size.replace('M','')
        size = float(size) * 1048576
        return(size)
    elif 'k' in size:
        size = size.replace('k','')
        size = float(size) * 1024
        return(size)
    else:
        return None


# In[7]:


playstoredata['Size'] = playstoredata['Size'].apply(lambda siz: Size(siz))
playstoredata.dropna(axis =0,inplace = True)


# In[8]:


# creating dummy variable for the non numerci data
category = pd.get_dummies(playstoredata['Category'],prefix='Ct')
Type = pd.get_dummies(playstoredata['Type'],prefix='ty')
content_rating = pd.get_dummies(playstoredata['Content Rating'],prefix='Cr')
Geners_rating = pd.get_dummies(playstoredata['Genres'],prefix = 'Grs')


# In[9]:


playstoredata['Rating'] = playstoredata['Rating'].apply(lambda rat:round(rat))


# In[10]:


# combining all the modified variables
Target = playstoredata['Rating'].values
Reviews = playstoredata['Reviews']
Installs = playstoredata['Installs']

final_df = [category,Type,content_rating,Geners_rating,Reviews,Installs]
Concated_df = pd.concat(final_df,axis =1)


# In[11]:


# scaling values to make sure values large values don't effect the small values
from sklearn.preprocessing import MinMaxScaler
MinMScaler = MinMaxScaler()
transoformed_df = MinMScaler.fit_transform(Concated_df)


# In[12]:


# separating data
CX_train,CX_test,Cy_train,Cy_test = train_test_split(transoformed_df,Target,random_state = 3,test_size = 0.2)


# In[13]:


# importing algorithms and validation libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics


# In[14]:


DecisionTree = DecisionTreeClassifier(criterion='entropy')
DecisionTree.fit(CX_train,Cy_train)
DecisionTree_Preds = DecisionTree.predict(CX_test)


# In[15]:


accuracy_score(Cy_test,DecisionTree_Preds)


# In[16]:


RadndomClassifier = RandomForestClassifier()
RadndomClassifier.fit(CX_train,Cy_train)
Random_preds = RadndomClassifier.predict(CX_test)
accuracy_score(Cy_test,Random_preds)


# In[17]:


RandomCls = RandomForestClassifier(n_estimators=1000)
RandomCls.fit(CX_train,Cy_train)
RandomCls_preds = RandomCls.predict(CX_test)
accuracy_score(Cy_test,RandomCls_preds)


# In[18]:


# This algortihm gives the best accuracy for this data
DTCost = DecisionTreeClassifier(max_depth=6 ,criterion='entropy')
DTCost.fit(CX_train,Cy_train)
DTcost_pred = DTCost.predict(CX_test)
accuracy_score(Cy_test,DTcost_pred)

