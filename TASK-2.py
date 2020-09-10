#!/usr/bin/env python
# coding: utf-8

# # Presented by-Harshada Suresh Sonawane

# # Task-2#
# To Explore Supervised Machine Learning
# 
# Given Task-What will be predicting score if a student study for 9.25 hrs in a day?

# # Simple Linear Regression#
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# In[6]:


# Import Libraries#
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


# Reading data #
path = "http://bit.ly/w-data"
data_set = pd.read_csv(path)
print("data_set")

data_set.head(10)


# # Basic Statistics of the data#

# In[8]:


print(data_set.describe())


# # Visualization#

# In[10]:


# Plotting the distribution of scores
data_set.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# **From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.**

# In[12]:


import seaborn as sns
corr=data_set.corr()
sns.heatmap(corr,annot=True,cmap='winter')
plt.show()


# In[13]:


#plotting the distribution of scores
sns.distplot(data_set['Scores'])


# # Preparing the data#
# 

# In[15]:


X = data_set.iloc[:, :-1].values  
y = data_set.iloc[:, 1].values  


# # Splitting the data into training and testing#

# In[16]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# In[17]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[18]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# ##Making Predictions#

# In[ ]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[ ]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[ ]:


# test with our data
hours = 9.25
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# ##Evaluating the model##
# 
# 

# In[ ]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 

