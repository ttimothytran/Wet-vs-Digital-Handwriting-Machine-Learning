#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import pickle


# In[2]:


DIRECTORY = r'C:\\Users\\thyti\\Desktop\\Wet vs Digital'
CATEGORIES = ['wet', 'digital']


# In[7]:


IMG_SIZE = 100

data = []

for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)
    label = CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        data.append([img_arr, label])


# In[8]:


len(data)


# In[9]:


random.shuffle(data)


# In[12]:


X = []
Y = []

for features, labels in data:
    X.append(features)
    Y.append(labels)


# In[13]:


X = np.array(X)
Y = np.array(Y)


# In[17]:


pickle.dump(X, open('X.pkl', 'wb'))
pickle.dump(Y, open('Y.pkl', 'wb'))


# In[ ]:




