
# coding: utf-8

# In[1]:


# importing libraries
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split


# In[2]:


playstoredata = pd.read_csv('googleplaystore.csv')


# In[3]:


print(playstoredata.isnull().sum())


# In[4]:


playstoredata.dropna(axis =0,inplace = True)


# In[5]:


def Install(installs):
    installs = installs.strip('+')
    installs = installs.replace(',','')
    return int(installs)


# In[6]:


playstoredata['Installs'] = playstoredata['Installs'].apply(lambda ins:Install(ins))
playstoredata.dropna(axis =0,inplace = True)


# In[7]:


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


# In[8]:


playstoredata['Size'] = playstoredata['Size'].apply(lambda siz: Size(siz))
playstoredata.dropna(axis =0,inplace = True)


# In[9]:


category = pd.get_dummies(playstoredata['Category'],prefix='Ct')
Type = pd.get_dummies(playstoredata['Type'],prefix='ty')
content_rating = pd.get_dummies(playstoredata['Content Rating'],prefix='Cr')
Geners_rating = pd.get_dummies(playstoredata['Genres'],prefix = 'Grs')


# In[10]:


playstoredata['Rating'] = playstoredata['Rating'].apply(lambda rat:round(rat))


# In[11]:


print(playstoredata['Rating'].value_counts())


# In[12]:


Target = playstoredata['Rating'].values
Reviews = playstoredata['Reviews']
Installs = playstoredata['Installs']

final_df = [category,Type,content_rating,Geners_rating,Reviews,Installs]
Concated_df = pd.concat(final_df,axis =1)


# In[13]:


print(Concated_df.shape)


# In[14]:


from sklearn.preprocessing import MinMaxScaler
MinMScaler = MinMaxScaler()
transoformed_df = MinMScaler.fit_transform(Concated_df)


# In[15]:


CX_train,CX_test,Cy_train,Cy_test = train_test_split(transoformed_df,Target,random_state = 3,test_size = 0.2)


# In[16]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics


# In[17]:


DecisionTree = DecisionTreeRegressor()
DecisionTree.fit(CX_train,Cy_train)
DecisionTree_Preds = DecisionTree.predict(CX_test)


# In[18]:


mean_squared_error(Cy_test,DecisionTree_Preds)


# In[19]:


RadndomClassifier = RandomForestRegressor()
RadndomClassifier.fit(CX_train,Cy_train)
Random_preds = RadndomClassifier.predict(CX_test)
mean_squared_error(Cy_test,Random_preds)


# In[20]:


RandomCls = RandomForestRegressor(n_estimators=1000)
RandomCls.fit(CX_train,Cy_train)
RandomCls_preds = RandomCls.predict(CX_test)
mean_squared_error(Cy_test,RandomCls_preds)


# In[21]:


# Decision tree regressor best fits this data as its error is less than other algorithms
DTCost = DecisionTreeRegressor(max_depth=6)
DTCost.fit(CX_train,Cy_train)
DTcost_pred = DTCost.predict(CX_test)
mean_squared_error(Cy_test,DTcost_pred)

