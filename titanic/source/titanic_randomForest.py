import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

# Combine train and test data for feature engineering
test['Survived'] = -1  # Placeholder for test set
combined = pd.concat([train, test])

# Fill missing values
combined['Age'] = combined['Age'].fillna(combined['Age'].median())
combined['Fare'] = combined['Fare'].fillna(combined['Fare'].median())

# Encode categorical variables for Random Forest
combined['Sex'] = combined['Sex'].map({'male': 0, 'female': 1})
combined['Embarked'] = combined['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Split data back into train and test sets
train = combined[combined['Survived'] != -1]
test = combined[combined['Survived'] == -1]

# Select features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = train[features]
y = train['Survived']
X_test = test[features]

# Train model
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

# Make predictions on the test set
test_predictions = rf.predict(X_test)

# Prepare submission file
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': test_predictions})
print(output)

