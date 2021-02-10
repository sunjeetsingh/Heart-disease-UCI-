#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv(r"C:\Users\HP\Desktop\datasets\heart.csv")


# In[3]:


df.head()


# In[4]:


df = df.rename(columns = {"age":"Age", "sex": "Sex", "cp":"Chest_pain", "trestbps":"Resting_BP", "chol":"Cholestrol", "fbs":"Fasting_blood_sugar", "restecg":"ECG_result", "thalach":"Max_heartrate", "exang":"Exercise_induced_agina", "oldpeak":"ECG difference", "ca":"Blocked vessels","slope":"Slope", "thal":"Thalassemia", "target":"Target"})
#renaming columns


# In[5]:


df.head()


# In[6]:


df1 = df.copy()
df2 = df.copy() #Making copies for later use


# In[88]:


df.info()


# we can see that the columns don't have any null values and the datatype is int and float, so that we don't have to deal with null values and change data types
# 

# In[96]:


df.describe()


# we can see that there is huge difference between the 50% and 75% of Cholestrol and ECG_difference which suggests that there are outliers, to confirm this, we will plot a boxplot

# In[112]:


sns.boxplot(x = df["Cholestrol"])


# In[111]:


sns.boxplot(x = df["ECG difference"])


# As evident from the above 2 plots, there are certain outliers in both cholestrol and ECG difference which can be dealt with while modelling

# In[94]:


df.corr()
plt.figure(figsize = (20,10))
sns.heatmap(df.corr(), annot = True)


# # Univariate Analysis

# In[113]:


plt.hist(df["Sex"])
df["Sex"].value_counts()


# we can see that there are more males in our obeservation than females

# In[7]:


plt.figure(figsize = (5,5))
plt.title("Age")
plt.hist(df["Age"])


# We can see that we have a variety of cases ranging from 30 to approximately 80 years of age, which gives us a good range and varitey of cases

# In[8]:


plt.title("Cholesterol")
plt.hist(df["Cholestrol"])


# the cholesterol lies between 200 to 400, most people have their cholesterol between 200-300 range

# In[9]:


plt.title("Blood Pressure")
plt.hist(df["Resting_BP"])


# the blood pressure lies between 100-180, the most common BP is between 120-140

# # Bivariate Analysis

# In[10]:


plt.figure(figsize = (10,5))
sns.countplot(x = df["Age"], hue = df["Sex"])


# we can see that there are more males than females in the data

# In[21]:


plt.figure(figsize = (30,5))
sns.countplot(x = df["Age"], hue = df["Fasting_blood_sugar"])


# fasting blood sugar relationship with Age

# In[27]:


plt.figure(figsize = (10,5))
sns.countplot(x = df["Age"], hue = df["Target"] )


# we can see that maximum heart attacks happen during the age of 40-60

# In[30]:


plt.figure(figsize = (15,5))
sns.countplot(x = df["Age"], hue = df["Thalassemia"])


# we can see that before 50, Type 2 is more common and after 55, both type 2 and type 3 are present, type 1 is more evident in ages 55-65 and there are little traces of type 0

# In[33]:


sns.countplot(x = df["Thalassemia"], hue = df["Target"])


# we can see that majority of positive targets are have type 2 thalassemia

# In[36]:


sns.countplot(x = df["Blocked vessels"], hue = df["Target"])


# # Gathering some more data

# In[54]:


a = df.groupby(["Sex", "Age"])["Target"].count().reset_index().sort_values(by = "Target", ascending = False)
a["Sex"].replace({1:"Male", 0:"Female"}, inplace = True)
a.head(20).style.background_gradient(cmap = "Greys")


# we can see from the above data that older males are more susceptible to heart attacks than females 

# In[87]:


a = df.groupby(["Age" ,"Chest_pain"])["Target"].count().reset_index().sort_values(by = "Target", ascending = False)
#a["Sex"].replace({1:"Male", 0:"Female"}, inplace = True)
a.head(20).style.background_gradient(cmap = "Greys")


# we can see that males above 50 having chest pains of type 0 are more susceptible to heart attacks than any other types

# In[ ]:





# In[ ]:





# In[ ]:




