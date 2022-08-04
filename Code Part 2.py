#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle


# In[2]:


X = pickle.load(open('X.pkl', 'rb'))
Y = pickle.load(open('Y.pkl', 'rb'))


# In[5]:


X = X/255


# In[7]:


X.shape


# In[8]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# In[9]:


model = Sequential()

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))

model.add(Dense(2, activation = 'softmax'))


# In[10]:


model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[11]:


model.fit(X, Y, epochs=5, validation_split=0.1)


# In[ ]:




