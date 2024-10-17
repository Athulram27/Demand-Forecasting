#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("C:/Users/athul/Downloads/python/Demand Forecasting/train.csv")


# In[3]:


df[['day','month','year']] = df['week'].str.split('/',expand = True)
df =df.drop('week',axis = 1)


# In[4]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X, y = df.drop('units_sold', axis = 1), df['units_sold']

X = X.dropna()
y = y[X.index] 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)


# In[5]:


model = RandomForestRegressor(n_jobs = -1)
model.fit(X_train, y_train)


# In[6]:


model.score(X_test, y_test)


# In[7]:


from sklearn.metrics import mean_squared_error
import numpy as np

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")


# In[8]:


df.units_sold.hist()


# In[9]:


import matplotlib.pyplot as plt

plt.scatter(y_pred, y_test)
plt.plot(np.linspace(y_pred.min(), y_pred.max()), np.linspace(y_test.min(), y_test.max()), color = 'red')


# In[10]:


df.hist(figsize = (15,8))
plt.show()


# In[11]:


df =df.drop('record_ID', axis=1)


# In[12]:


len(df.store_id.unique())


# In[13]:


len(df.sku_id.unique())


# In[14]:


df = df[df.units_sold < df.units_sold.quantile(0.99)]


# In[15]:


df.units_sold.hist(bins = 20)


# In[16]:


df = df.join(pd.get_dummies(df.store_id, prefix = 'store')).drop('store_id', axis = 1)


# In[17]:


df = df.join(pd.get_dummies(df.sku_id, prefix = 'sku')).drop('sku_id', axis = 1)


# In[18]:


model = RandomForestRegressor(n_jobs = -1)

X, y = df.drop('units_sold', axis = 1), df['units_sold']

X = X.dropna()
y = y[X.index] 

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[19]:


model.fit(X_train, y_train)

score = model.score(X_test, y_test)  
print(f"R^2 Score: {score}")

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")


# In[20]:


plt.scatter(y_pred, y_test)
plt.plot(np.linspace(y_pred.min(), y_pred.max()), np.linspace(y_test.min(), y_test.max()), color = 'red')


# In[26]:


# hyper parameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [10,20],
    'min_samples_split': [2,3]
}

model = RandomForestRegressor(n_jobs=-1)

grid_search = GridSearchCV(model, param_grid, verbose = 2, cv = 3)

grid_search.fit(X,y)



# In[29]:


best_model = grid_search.best_estimator_
grid_search.best_params_


# In[30]:


best_model.score(X_test, y_test)


# In[ ]:




