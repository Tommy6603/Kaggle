import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime as dt
from datetime import datetime, timedelta
import random
from statsmodels.formula.api import logit
# Supervised Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestRegressor
# Boosts Models
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor
# Train / Test and Score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
# UnSupervised Models
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.tree import plot_tree
# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load the modules and the datasets
train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean()) # There was 1 na to fill
gender_data = pd.read_csv("./data/gender_submission.csv")

# Show train data 
train_data.head()

# Count how many people survived and the % 
print(f"A total of {train_data['Survived'].sum()} people survived, out of {len(train_data)}. That's the {round(train_data['Survived'].sum()/len(train_data)*100)}%")

# Plot logical regression on Fare vs Survived
plt.figure(figsize = (4,3))
plt.title('Fare vs Survived')
sns.regplot(x='Fare', y='Survived', data=train_data, ci=None, logistic=True)
plt.show()


# As we can see, the highest the Fare the highest the chance of survived. 
# A question arises on that Fare 500. Is it an outliner? Let's find out!
print(train_data[train_data['Fare']>400])

# Those 3 people paid much more than others (and all of them survived)
# It seems that Fare may be a good estimator of Survived chance, but let's explore further!


# Plot logical regression on Age vs Survived
plt.figure(figsize = (4,3))
plt.title('Age vs Survived')
sns.regplot(x='Age', y='Survived', data=train_data, ci=None, logistic=True)
plt.show()

# Interestingly, the oldest the age, the lower the percentage of Survived
# The problem is that Age has lots of NA.. so it won't be used


# Plot logical regression on Ticket Class vs Survived
plt.figure(figsize = (4,3))
plt.title('Ticket Class vs Survived')
sns.regplot(x='Pclass', y='Survived', data=train_data, ci=None, logistic=True)
plt.show()

# The better the class, the higher the chance of Survived.


# Change 'male' with 1 and 'female' with 0 in order to create a plot
train_data['Sex'] = train_data['Sex'].replace(['male', 'female'], [1, 0])

# Plot logical regression on Sex vs Survived
plt.figure(figsize = (4,3))
plt.title('Sex vs Survived')
sns.regplot(x='Sex', y='Survived', data=train_data, ci=None, logistic=True)
plt.show()

# As we can see, female (0) had an higher chance of Survived


# What's the relationship between Ticket Class and Fare?
avg_first = train_data[train_data['Pclass'] == 1]['Fare'].mean()
avg_second = train_data[train_data['Pclass'] == 2]['Fare'].mean()
avg_third = train_data[train_data['Pclass'] == 3]['Fare'].mean()
print(f"The average Fare of First Ticket Class is: {round(avg_first,)}")
print(f"The average Fare of Second Ticket Class is: {round(avg_second)}")
print(f"The average Fare of Third Ticket Class is: {round(avg_third)}")

# Now we know the relationships between Fare, Age, Class, Sex.. 
# It's time to create a Machine-Learning Model!

SEED = 1
X_train = train_data[['Fare', 'Sex', 'Pclass']].copy()
# Change 'male' with 1 and 'female' with 0
X_train['Sex'] = X_train['Sex'].astype('category')
X_train['Sex'] = X_train['Sex'].cat.rename_categories({'male': 1, 'female': 0}).astype(int)
y_train = train_data['Survived']

X_test = test_data[['Fare', 'Sex', 'Pclass']].copy()
# Change 'male' with 1 and 'female' with 0
X_test['Sex'] = X_test['Sex'].astype('category')
X_test['Sex'] = X_test['Sex'].cat.rename_categories({'male': 1, 'female': 0}).astype(int)
y_test = gender_data['Survived']


# Instantiate Logistic Regression
lr = LogisticRegression(random_state=SEED)

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy of the model is:", round(accuracy*100,2), "%")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# A super high %! And the Confusion Matrix shows only 2 errors.

# Plot logical regression on Data vs Predictions
plt.figure(figsize = (4,3))
plt.title('Data vs Predictions')
sns.regplot(x=y_test, y=y_pred, ci=None, logistic=True)
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})
# Let's check another proof of the similarities between predictions and data
print(output.head())
print(gender_data.head())

# This is all.
# Save the output and submit!
output.to_csv('submission.csv', index=False)
print("\nYour submission was successfully saved!")



