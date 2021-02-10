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


df = df.rename(columns = {"age":"Age", "sex": "Sex", "cp":"Chest_pain", "trestbps":"Resting_BP", "chol":"Cholestrol", "fbs":"Fasting_blood_sugar", "restecg":"ECG_result", "thalach":"Max_heartrate", "exang":"Exercise_induced_agina", "oldpeak":"ECG_difference", "ca":"Blocked vessels","slope":"Slope", "thal":"Thalassemia", "target":"Target"})


#  Changing column names to understand data more

# In[4]:


df.head()


# In[5]:


df1 = df.copy()
df2 = df.copy()


# In[6]:


df.info()


# The data doesn't have any null values and datatypes are also workable, hence we can now start implementing ML models

# In[7]:


target = df["Target"]
df.drop(columns = "Target", inplace = True)


# In[8]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[12]:


xtrain, xtest, ytrain, ytest = train_test_split(df,target, test_size=0.3, random_state = 0)


# In[13]:


scaler = StandardScaler()


# In[14]:


xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)


# # Logistic Regression

# In[23]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[34]:


parameters = {'C' :[0.1, 1, 10, 100,]}
gridsearch = GridSearchCV(model, parameters, scoring = 'accuracy', cv = 10)
gridsearch.fit(xtrain, ytrain)
gridsearch.best_params_


# In[35]:


Lr= LogisticRegression(C = 0.1)
Lr.fit(xtrain, ytrain)
yhat = Lr.predict(xtest)


# In[39]:


print("accuracy_score of the model",accuracy_score(ytest,yhat))
print("confusion matrix for the model",confusion_matrix(ytest, yhat))
a = confusion_matrix(ytest,yhat)
sns.heatmap(a, annot = True)


# # K-Nearest Neighbor 

# In[40]:


from sklearn.neighbors import KNeighborsClassifier


# In[41]:


KNN_accuracy = []
for k in range(1,10):
    KNN = KNeighborsClassifier(n_neighbors = k)
    KNN.fit(xtrain, ytrain)
    yhat = KNN.predict(xtest)
    print(accuracy_score(ytest,yhat))
    KNN_accuracy.append(accuracy_score(ytest,yhat))


# In[42]:


plt.plot(range(1,10), KNN_accuracy)
plt.xlabel("n_neighbours")
plt.ylabel("Accuracy")


# k = 6 gives the highest accuracy, hence we will use n_neighbors = 6

# In[45]:


realKNN = KNeighborsClassifier(n_neighbors = 6)
realKNN.fit(xtrain, ytrain)
yhat = realKNN.predict(xtest)
print(accuracy_score(ytest,yhat))


# In[47]:


sns.heatmap(confusion_matrix(ytest,yhat), annot = True)


# # Decision Tree

# In[48]:


from sklearn.tree import DecisionTreeClassifier


# In[97]:


model_decision_tree1 = DecisionTreeClassifier(criterion = "gini", random_state = 108, max_depth = 10, min_samples_leaf = 3 )


# In[98]:


model_decision_tree1.fit(xtrain,ytrain)
predicted_decision_tree1 = model_decision_tree1.predict(xtest)


# In[103]:


print("accuracy score is -",accuracy_score(ytest,predicted_decision_tree1))
sns.heatmap(confusion_matrix(ytest,predicted_decision_tree1), annot = True)


# In[104]:


from sklearn.svm import SVC


# In[105]:


support_vector_model = SVC()


# In[106]:


support_vector_model.fit(xtrain,ytrain)
SVC_prediction = support_vector_model.predict(xtest)


# In[107]:


print("accuracy score is -",accuracy_score(ytest,SVC_prediction))
sns.heatmap(confusion_matrix(ytest,SVC_prediction), annot = True)


# In[ ]:




