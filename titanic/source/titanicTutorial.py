# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Any results you write to the current directory are saved as output.


train_data = pd.read_csv("./data/train.csv")
train_data.head()

test_data = pd.read_csv("./data/test.csv")
test_data.head()


women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)


men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"] # predict if suvived or non-survived

features = ["Pclass", "Sex", "SibSp", "Parch"] # Select Feature value
X = pd.get_dummies(train_data[features]) 
# get_dummies : category variable to dummy variable. (separate column of sex to sex_male and sex_female→ you can see it binary)
X_test = pd.get_dummies(test_data[features])

print(train_data)


"""
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
print(output)
"""


