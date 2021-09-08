#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[14]:


df = pd.read_csv('D:/Workspace/forecast/cairns.csv',parse_dates = [3])

df = df.iloc[:, [3,4]]
df = df.set_index('valid_start')
df.index.freq='H'


# In[7]:


#df['periods'] = pd.date_range(start = '2016-04-30 14:00:00', periods = len(df), freq='H')
#df = df.set_index('periods')


# In[9]:


train, test = df[:17386], df[17386:]


# In[10]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[12]:


scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# In[14]:


from keras.preprocessing.sequence import TimeseriesGenerator


# In[15]:


# define generator
n_input = 168
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)


# In[20]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# In[24]:


# define model
model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[ ]:


# fit model
model.fit(generator,epochs=10)


# In[ ]:


loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)


# In[44]:


test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    # get the prediction value for the first batch
    current_pred = model.predict(current_batch)[0]
    
    # append the prediction into the array
    test_predictions.append(current_pred) 
    
    # use the prediction to update the batch and remove the first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[47]:


true_predictions = scaler.inverse_transform(test_predictions)


# In[50]:


test['Predictions'] = true_predictions


# In[54]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(test['PRCP'],test['Predictions']))
print(rmse)


# In[ ]:




