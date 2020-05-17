#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score


# In[5]:


# It has around 303 patients collected from the Cleveland Clinic Foundation. 
url = "Heart_Diagnosis.csv"

# the names will be the names of each column in our pandas DataFrame
names = ['age',
        'sex',
        'cp',
        'trestbps',
        'chol',
        'fbs',
        'restecg',
        'thalach',
        'exang',
        'oldpeak',
        'slope',
        'ca',
        'thal',
        'class']
cleveland = pd.read_csv(url, names=names)
print('Shape of DataFrame: {}'.format(cleveland.shape))


# In[6]:


# remove the missing data (indicated with a '?')
data = cleveland[~cleveland.isin(['?'])]
print(data.loc[280:])


# In[7]:


data = data.dropna(axis=0)
print(data.loc[280:])


# In[8]:


print(data.shape)
print(data.dtypes)


# In[9]:


data = data.apply(pd.to_numeric)
print(data.dtypes)


# In[10]:


data.hist(figsize = (12, 12))
plt.show()


# In[11]:


# create X and Y datasets for training
X = np.array(data.drop(['class'],1))
y = np.array(data['class'])


# In[12]:


# split training and testing data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)


# In[14]:


Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print(Y_train.shape)
print(Y_train[:10])


# In[15]:


# define a function to buils keras model
def create_model():
    #create
    model = Sequential()
    model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, activation = 'softmax'))
    
    #compile
    adam = Adam(lr = 0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


# In[23]:


model = create_model()
print(model.summary())

#fit the model to the training data
model.fit(X_train, Y_train, epochs=100, batch_size=10, verbose = 1)


# In[18]:


Y_train_binary = y_train.copy()
Y_test_binary = y_test.copy()
Y_train_binary[Y_train_binary > 0] = 1
Y_test_binary[Y_test_binary > 0] = 1

print(Y_train_binary[:20])


# In[20]:



# define new kaeras model for binary classification
def create_binary_model():
    model = Sequential()
    model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model
binary_model = create_binary_model()
print(binary_model.summary())


# In[21]:


# fit the binary model on the training data
binary_model.fit(X_train, Y_train_binary, epochs=100, batch_size=10, verbose=1)


# In[24]:


# generate classification report using predictions for categorical model
categorical_pred = np.argmax(model.predict(X_test), axis=1)


# In[25]:


print("Results for categorical model")
print(accuracy_score(y_test, categorical_pred))
print(classification_report(y_test, categorical_pred))


# In[26]:


# generate classification report using prediction for binary model
binary_pred = np.round(binary_model.predict(X_test)).astype(int)
# astype is used to create copy of the array, cast to a specified type(int, here)
print('Results for Binary Model')
print(accuracy_score(Y_test_binary, binary_pred))
print(classification_report(Y_test_binary, binary_pred))


# In[ ]:





# In[ ]:




