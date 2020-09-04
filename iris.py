#!/usr/bin/env python
# coding: utf-8

# Iris Flower Classification
# 
# The problem we're trying to solve in this project is known as the Iris Flower Dataset. It is a dataset that consists of 150 records (flowers) classified into three different species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: sepal length, sepal width, petal length, petal width.

# **Our goal is to find out to which species a flower belongs based on those four features.

# We are going to be dealing with a new library called scikit-learn (sklearn) to get the data and to perform some operations.
# 

# # Step 1: Get the Data

# In[4]:


from sklearn.datasets import load_iris

# Let's create a variable to hold the dataset
iris = load_iris()


# # Step 2: Explore the data

# In[5]:


# First we need to import some important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


#The dataset is provided as a python dictionary object, let's check its keys...
iris.keys()


# Let's create a DataFrame object out of the dictionary we have:

# In[32]:


data = pd.DataFrame(iris['data'], columns=iris['feature_names'])
data['species'] = iris['target']


# Now, we can use pandas methods to explore the dataset

# In[8]:


data.head()


# In[9]:


data.info()


# In[10]:


data.describe()


# Let's draw some graphs using Matplotlib to understand the relationships between the four properties of an Iris flower and its species:

# The first one here describes the relationship between Sepal length (x-axis), Petal length (y-axis), and the three species:

# purple ---> Setosa

# green ---> Versicolor

# yellow ---> Verginica

# In[11]:


plt.scatter(data['sepal length (cm)'] , data['petal length (cm)'], c=data['species'])
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()


# In[13]:


plt.scatter(data['sepal width (cm)'] , data['petal width (cm)'], c=data['species'])
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()


# In[14]:


plt.scatter(data['sepal length (cm)'] , data['petal width (cm)'], c=data['species'])
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()


# In[15]:


plt.scatter(data['sepal width (cm)'] , data['petal length (cm)'], c=data['species'])
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()


# Boxplots are statistical graphs that could give us great knowledge about distinct categories
# since we have 3 different species, it is a good idea to take a look at them.
# Boxplot Interpretation:

# First edge of the rectangle: 25% Percentile

# Orange line: 50% Percentile (Median)

# Last edge of the rectangle: 75% Percentile

# Circles: Outliers

# In[162]:


# First we need to have our data separated into 3 lists. One for each species.
grouped_data = [data['sepal length (cm)'][temp*50:(temp+1)*50] for temp in range(0,3)]
plt.boxplot(grouped_data)
plt.title('Boxplot for Sepal Length ')
plt.xlabel('Species')
plt.show()


# In[163]:


grouped_data = [data['sepal width (cm)'][temp*50:(temp+1)*50] for temp in range(0,3)]
plt.boxplot(grouped_data)
plt.title('Boxplot for Sepal Width ')
plt.xlabel('Species')
plt.show()


# In[164]:


grouped_data = [data['petal length (cm)'][temp*50:(temp+1)*50] for temp in range(0,3)]
plt.boxplot(grouped_data)
plt.title('Boxplot for Petal Length ')
plt.xlabel('Species')
plt.show()


# In[165]:


grouped_data = [data['petal width (cm)'][temp*50:(temp+1)*50] for temp in range(0,3)]
plt.boxplot(grouped_data)
plt.title('Boxplot for Petal Width ')
plt.xlabel('Species')
plt.show()


# # Step 3: Training a Machine Learning model

# We will start to use some functions out of the Scikit-Learn library to do 3 main tasks:
# - Split the data into (train set) and (test set)
# - Select and train a model
# - Evaluate the model

# 1. Splitting the data:
#   

# Split the data to train and test subsets... just so you can evaluate your model after training. It doesn't make
#   sense to evaluate the model using the data that it has been trained on.
#   
#   SK-Learn has many built-in functions to do that. We are going to be using one of them, called train_test_split, to 
#   split our data and its labels into two different sets

# In[33]:


# Remove the target ('species') from the data
labels = data['species'].copy()
data.drop('species', axis=1, inplace=True)


# In[149]:


from sklearn.model_selection import train_test_split 
train, test, train_labels, test_labels = train_test_split(data, labels, random_state=0)

# train: A variable that will be holding the values of the features of each flower (in the train subset)
# train_labels: A variable that will be holding the target values of the corresponding flowers (in the train subset)
# test: A variable that will be holding the values of the features of each flower (in the test subset)
# test_labels: A variable that will be holding the target values of the corresponding flowers (in the test subset)


# 2. Selecting and training a model

# First we'll go with an algorithm called KNN (K Nearest Neighbors).
# Simply, what it does is that it looks for the closest K (an integer) instances to the desired instance... and consider the new sample to belong to the class of the majority of those neighbors.
# 
# The image below desscribes the process

# ![KNN_final_a1mrv9.png](attachment:KNN_final_a1mrv9.png)

# In[150]:


# Import the model
from sklearn.neighbors import KNeighborsClassifier

# Let's create an object from the class KNeighborsClassifier. Just for simplicity, set the number of neighbors to 1
knn = KNeighborsClassifier(n_neighbors=1)

# Now we have to train the model on the training data, and that is simply done by the (fit) function.
knn.fit(train, train_labels)

# WELL DONE, now the model is ready to predict the targets of new samples. Just use the (predict) function.
predictions = knn.predict(test)


# In[161]:


# Let's see the predictions of our model and compare them with the real targets.
print(list(test_labels))
print(list(predictions))


# # Evaluation

# In[125]:


from sklearn.metrics import classification_report


# Here are some statistical measures that describes the accuracy of our model

# In[159]:


print('Accuracy =', knn.score(test, test_labels), '\n')
print(classification_report(test_labels, predictions))


# Let's try another algorithm.
# Decision tree classifier

# In[160]:


from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(train,train_labels)

predictions = tree.predict(test)

print('Accuracy =', knn.score(test, test_labels), '\n')
print(classification_report(test_labels, predictions))
